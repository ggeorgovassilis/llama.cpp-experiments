# Prefill parallelisation in layer-split models: mechanism analysis

Status: analysis complete. Tracked by: issue #8.
Companion analysis (decode path): `pipeline-parallelism-analysis.md`.

---

## 1. Summary

Prefill in a layer-split model is not one big batch. The server packs prompt tokens
into batches of up to `n_batch` (default 2048), and `llama_decode` further splits each
batch into `n_ubatch`-sized sub-batches (default 512). Each sub-batch is built as its
own graph and split into the same 4-5 backend splits as a decode step
(embedding/GPU0 layers -> GPU1 -> GPU2 -> GPU3 -> logits). Prefill is therefore a
stream of `n_ubatch`-sized sub-batches through the same serial layer pipeline, not a
single parallelisable batch.

The three open questions resolve as follows:

- Prefill is chunked: a sequence of `n_ubatch`-sized sub-batches, each with the same
  4-5-split layer pipeline as decode.
- Consecutive prefill sub-batches do NOT overlap in the current code. The server does
  skip its sync while `has_output` is false, but the graph-reuse path in
  `llama_context::process_ubatch` re-introduces a full `ggml_backend_sched_synchronize`
  between same-shape sub-batches whenever `pipeline_parallel` is on. That barrier is
  the actual blocker, not the server.
- The bottleneck is compute (matmul), not cross-GPU transfer.

## 2. How prefill is represented in the graph splits

Two levels of batching:

1. Server level (`tools/server/server-context.cpp`). The prompt tokens are added to a
   `llama_batch` until it reaches `n_batch` (`llama_n_batch`). `decode()` then submits
   one `llama_decode` per filled batch. A prompt longer than `n_batch` spans several
   `llama_decode` calls.

2. Context level (`src/llama-context.cpp`, `llama_context::decode`, ~line 1784). Each
   `llama_decode` splits its batch into `n_ubatch`-sized `llama_ubatch`es and runs
   `process_ubatch` per ubatch in a `do/while` loop. Every ubatch is a separate graph
   (`model.build_graph`), allocated and computed independently.

Each prefill ubatch builds the same graph topology as a decode step (all layers, one
attention pass per token), then `ggml_backend_sched_split_graph` partitions it into
contiguous backend splits. For a 4-GPU layer split this is ~5 splits: token embedding +
early layers on GPU0, middle layers on GPU1/GPU2, late layers + logits on GPU3 (and a
CPU/logits tail depending on offload). The ubatch differs from decode only in the
number of tokens per split (512 vs 1), so the split *count* is the same but the
matmuls inside each split are larger.

## 3. Why consecutive prefill sub-batches do not overlap

There are two candidate sync points between consecutive prefill ubatches.

### 3.1 The server sync (does not fire for prefill)

`tools/server/server-context.cpp`, `decode()` (~line 3673):

```c
bool has_output = false;
for (int i = off; i < off + batch_view.n_tokens; ++i) {
    has_output |= batch.tokens[i].output;
}
...
queue_tasks.yield_to_queue([&]() {
    ret = llama_decode(ctx_tgt, batch_view);
    if (ret == 0 && has_output) {
        llama_synchronize(ctx_tgt);
    }
});
```

Prompt tokens are added with `output = false` (except embeddings). Only the final
prompt batch sets `output = true` on its last token (`batch.set_output(batch.size()-1,
true)` when the prompt is done). So intermediate prefill batches have `has_output ==
false` and skip the sync. The server does not block overlap between intermediate
prefill sub-batches.

### 3.2 The graph-reuse sync (the actual blocker)

`src/llama-context.cpp`, `process_ubatch` (~line 1411):

```c
if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
    // with pipeline parallelism, the previous graph_compute_async may still be running
    // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
    // that the previous compute is still reading.
    if (cparams.pipeline_parallel) {
        ggml_backend_sched_synchronize(sched.get());
    }
    n_reused++;
}
```

There are two graph-result slots, `gf_res_prev[n_outputs > 0]` (see `get_gf_res_prev`,
~line 2424): slot 0 for ubatches with no outputs (intermediate prefill), slot 1 for
ubatches with outputs (decode, and the final prefill ubatch). Consecutive same-size
prefill ubatches all land in slot 0, have identical topology, so `can_reuse` is true and
the graph is reused. When `pipeline_parallel` is on (which it is for a layer split with
all layers on GPU, per issue #2), the reuse path issues a full
`ggml_backend_sched_synchronize` before overwriting the inputs.

`ggml_backend_sched_synchronize` (`ggml/src/ggml-backend.cpp`, ~line 2033) synchronises
every backend: a full barrier. Because the graph is reused, `ggml_backend_sched_alloc_graph`
is not called again, so the `cur_copy`/`next_copy` rotation never advances and the same
buffer copy is reused every sub-batch. Net effect: consecutive prefill sub-batches are
strictly serialised, using one buffer copy.

This barrier fires both between ubatches *within* one `llama_decode` and between the last
ubatch of one `llama_decode` and the first of the next (same slot 0, same shape). So the
copies/events overlap that the scheduler provides (documented in
`pipeline-parallelism-analysis.md`, section 2.4) never materialises during prefill.

### 3.3 The one case where the copies overlap could fire

The copies rotation advances only when a graph is freshly allocated (topology changes,
`graph_reuse_disable`, or the switch between slot 0 and slot 1). The only prefill ubatch
that differs in shape is the final, partial one (e.g. 464 tokens out of 512). If
`LLAMA_GRAPH_REUSE_DISABLE=1` is set, every ubatch is rebuilt and reallocated, the
`cur_copy`/`next_copy` rotation advances each time, and consecutive ubatches use distinct
buffer/event slots - the precondition for pipeline overlap. This is the natural first
experiment.

## 4. Bottleneck: compute vs transfer

Prefill is compute-bound (matmul), not transfer-bound.

- Cross-GPU transfer is only the hidden-state activations at the layer boundaries, one
  copy per boundary. For a ~30B dense model (`n_embd` ~ 6-8k) with `n_ubatch = 512` in
  f32, that is roughly `512 * 8192 * 4 = 16 MB` per boundary, 3 boundaries ~ 48 MB per
  ubatch. KV cache is per-layer-local and is not transferred. At PCIe Gen3 x16 (~16 GB/s)
  this is a few ms.
- Compute is a matmul per token per layer, ~`2 * n_params` FLOPs/token. For 30B params
  and 512 tokens that is ~30 TFLOP per ubatch, spread serially across the 4 GM107 GPUs
  (each ~1-2 TFLOP effective for Q4_K_M int8 matmuls). Compute dominates by two orders of
  magnitude.

The under-utilisation observed in issue #2 is a pipeline (data dependency + the reuse
sync) problem, not a bandwidth problem. Parallelising prefill therefore means pipelining
across sub-batches (or requests), not reducing transfer volume.

## 5. Next steps (baseline phase, tracked in issue #8)

1. Confirm the reuse-sync is active: run a layer-split model on rig1 and check the
   startup log for `pipeline parallelism enabled` / `sched copies = 4`, then observe
   per-GPU SM during a single long prefill (expect the one-GPU-at-a-time signature, with
   a visible stall at each sub-batch boundary from the full synchronise).
2. Baseline prefill throughput (tokens/s) for a ~30B dense Q4_K_M model across the 4x
   M10, single request, using the `bench/` harness extended with a prompt-eval-only mode
   (`n_predict = 1`).
3. Re-run with `LLAMA_GRAPH_REUSE_DISABLE=1` and measure whether prefill throughput
   improves (tests the pipeline-overlap hypothesis without code changes).
4. If that shows promise, prototype a targeted fix (skip the reuse sync for the
   non-output prefill slot, or force fresh allocation per prefill ubatch) and re-measure.

Model selection follows issue #2's decision criteria (dense, clean 1:1 layer split);
candidates from the rig1 catalogue are NVIDIA-Nemotron-3.5-Lightning-30B or the
Qwen3.5-30B class, with Qwen3.5-9B-Q4_K_M as the light fallback already validated in #2.
