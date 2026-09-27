# Prefill parallelisation in layer-split models: mechanism analysis

Status: corrected after baseline measurement (see `prefill-parallelism-baseline.md`).
Tracked by: issue #8.
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
- Consecutive prefill sub-batches DO overlap in the current code. The server skips its
  sync while `has_output` is false, and the graph-reuse sync in
  `llama_context::process_ubatch` never fires during prefill: `can_reuse` is false
  because the attention KQ mask grows with every sub-batch (section 3.2). The
  `cur_copy`/`next_copy` overlap therefore materialises. Verified empirically in
  `prefill-parallelism-baseline.md` (reuse on/off identical; all 4 GPUs reach 100% SM).
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

## 3. Prefill sub-batches overlap (the reuse sync does not fire)

There are two candidate sync points between consecutive prefill ubatches. Only the
server one was ever in question, and it does not fire for prefill.

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

### 3.2 The graph-reuse sync (does not fire for prefill)

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
ubatches with outputs (decode, and the final prefill ubatch). For the reuse branch to
fire, `res->can_reuse(gparams)` must be true. For a dense model the attention input's
`can_reuse` ends in `can_reuse_kq_mask` (`src/llama-graph.cpp`, ~line 48):

```c
const auto n_kv = mctx->get_n_kv();
...
res &= (kq_mask->ne[0] == n_kv);
```

`n_kv` is the padded KV occupancy (`llama_kv_cache::get_n_kv`,
`src/llama-kv-cache.cpp`, ~line 1250) and grows with every prefill sub-batch: 512 ->
1024 -> 1536 -> ... . The graph built for sub-batch N has `kq_mask->ne[0] == n_kv(N)`,
which is never equal to `n_kv(N+1)`, so `can_reuse` is false and every prefill sub-batch
is freshly allocated (`res->reset()` + `ggml_backend_sched_alloc_graph`). The
`cur_copy`/`next_copy` rotation therefore advances every sub-batch, and the copies/events
overlap described in `pipeline-parallelism-analysis.md` section 2.4 materialises during
prefill.

This is the key asymmetry with decode: decode adds one token per step, so `n_kv` stays
inside the same 256-token padding block for many consecutive steps and the graph IS
reused (and the reuse sync fires). Prefill crosses a 256 boundary every sub-batch, so it
never reuses.

### 3.3 Disabling graph reuse is a no-op for prefill

Because `can_reuse` is already false for prefill, `LLAMA_GRAPH_REUSE_DISABLE=1` changes
nothing. This was the first experiment and it confirmed the prediction exactly
(`prefill-parallelism-baseline.md`): prefill tok/s is unchanged with reuse on vs off.

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

The under-utilisation observed in issue #2 is a pipeline problem (data dependency, plus
the reuse sync that only fires during decode), not a bandwidth problem. Prefill already
pipelines across sub-batches (verified in `prefill-parallelism-baseline.md`); the
remaining headroom is layer balance and pipeline bubbles, not transfer volume.

## 5. Result (baseline phase, tracked in issue #8)

The baseline was measured on rig1 and is recorded in
`prefill-parallelism-baseline.md`. Summary:

1. The reuse sync is NOT active during prefill (section 3.2). Startup shows
   `pipeline parallelism enabled` / `sched copies = 4`, and all 4 GPUs reach 100% SM
   simultaneously during a single long prefill - overlap is already happening.
2. Prefill tok/s (Qwen3.5-9B-Q4_K_M, layer split, K=1): 63 (512 tok), 70 (1024), 95
   (2048), 125 (4096), 154 (8192).
3. `LLAMA_GRAPH_REUSE_DISABLE=1` changes nothing (reuse never fires during prefill).
4. Conclusion: prefill is already parallelised across the 4 GPUs; there is no reuse
   barrier to remove. Remaining headroom is layer imbalance and pipeline bubbles, a
   separate question.
