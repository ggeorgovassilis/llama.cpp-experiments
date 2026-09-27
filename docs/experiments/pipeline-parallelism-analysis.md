# Pipeline parallelism in llama.cpp: mechanism, diagnostics, baseline

Status: complete (phases 1-5). Results: `pipeline-parallelism-baseline.md`.

Tracked by: issue #2.

---

## 1. Summary

The premise in the ticket is: with `--split-mode layer`, a model is spread across the
GPUs as contiguous layer slices, and each decode step runs the slices serially
(GPU0 -> GPU1 -> GPU2 -> GPU3), so only one GPU is busy at a time. This is essentially
correct for the stock `llama-server`, but the codebase is more nuanced than the premise
suggests.

Upstream llama.cpp (which this fork tracks on `master`) already contains a pipeline
parallelism implementation in the backend scheduler: a multiple-copies + CUDA-events
mechanism that can overlap the async compute of successive graph computations. That
mechanism exists and is enabled by default for the layer split, but the server's
synchronous generation loop defeats it in practice. The detail is documented below.

No experiment code exists yet. Branch `true-pipeline` is two documentation-only commits
(`initial setup`, `setup`) ahead of upstream `master`.

---

## 2. The exact mechanism

### 2.1 Layer placement (model load)

`--split-mode layer` (the default) places each layer's weights contiguously on one GPU.
`--tensor-split`/`-ts` controls the proportions (default: proportional to free VRAM);
`-ngl all` pushes every layer onto the GPUs. The KV cache for layer `l` lives on the GPU
that owns layer `l`. See `docs/multi-gpu.md`.

Result for a 40-layer model on 4 equal GPUs: GPU0 layers 1-10, GPU1 11-20, GPU2 21-30,
GPU3 31-40.

### 2.2 Backend list and scheduler

In `src/llama-context.cpp` (constructor, ~line 347) a backend is created per
`model.devices` (CUDA0..CUDA3), plus the CPU backend. The scheduler is created with:

```c
ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                       max_nodes, cparams.pipeline_parallel, cparams.op_offload);
```

So the scheduler sees `[CUDA0, CUDA1, CUDA2, CUDA3, CPU]`.

### 2.3 Graph splitting

`ggml_backend_sched_split_graph` (`ggml/src/ggml-backend.cpp`) assigns every node to the
backend that owns its tensors' buffers, then partitions the node list into contiguous
"splits": runs of consecutive nodes on the same backend. A dependency whose producer and
consumer are on different backends becomes a split "input", copied to the consumer's
backend at compute time. For a 4-GPU layer split this yields roughly 4-5 splits
(embedding/early layers on GPU0, ..., output/logits on the last backend/CPU).

### 2.4 The pipeline parallelism (copies + events)

The scheduler struct (`ggml_backend_sched`) carries:

```c
int n_copies;                                   // 1 normally, GGML_SCHED_MAX_COPIES (4) when parallel
int cur_copy, next_copy;                        // rotate on every graph allocation
ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES]; // CUDA events
```

`ggml_backend_sched_new(..., parallel, ...)` sets `n_copies = parallel ? 4 : 1` and
creates 4 events per backend when `parallel` is true.

`ggml_backend_sched_compute_splits` iterates the splits and, per split:

1. If the split has no cross-backend inputs, wait for the previous backend's event
   (`ggml_backend_event_synchronize`), because the allocator may reuse buffer regions.
2. Copy the split's input tensors to the split backend, waiting on that backend's event
   (`ggml_backend_event_wait`) before overwriting.
3. Compute the split asynchronously (`ggml_backend_graph_compute_async`, an async CUDA
   stream).
4. Record an event on the split backend (`ggml_backend_event_record`).

The `cur_copy`/`next_copy` rotation in `ggml_backend_sched_alloc_graph` means successive
graph computations use different buffers and different event slots, so the async compute
of graph N+1 can overlap the still-running graph N. This is the actual pipeline
parallelism: with 4 copies the work is double-buffered (quad-buffered) across decode
steps.

### 2.5 Enablement conditions

In `src/llama-context.cpp` (~line 428) `pipeline_parallel` is true iff all hold:

- `model.n_devices() > 1`
- `model.n_gpu_layers() > model.hparams.n_layer_all` (all layers on GPU)
- `model.split_mode() == LLAMA_SPLIT_MODE_LAYER`
- `cparams.offload_kqv` (default true)
- `!model.has_tensor_overrides()`
- every non-CPU backend reports `caps.async` and `caps.events` (CUDA does)

When true, the log prints `pipeline parallelism enabled`, and the reserve step prints
`sched copies = 4`. When false, `sched copies = 1`. This single log line is the key
runtime diagnostic (see section 3).

### 2.6 Why the server does not overlap during generation

`llama_context::decode()` (`src/llama-context.cpp`, line 1705) dispatches the graph
asynchronously and deliberately does NOT synchronize at the end
(`//synchronize();` is commented out). The server, however, synchronizes after every
decode that produces output:

```c
// tools/server/server-context.cpp, ~line 3676
queue_tasks.yield_to_queue([&]() {
    ret = llama_decode(ctx_tgt, batch_view);
    if (ret == 0 && has_output) {
        llama_synchronize(ctx_tgt);
    }
});
```

`llama_synchronize` -> `ggml_backend_sched_synchronize` -> synchronize all backends: a
full barrier. During generation every decode step has output, so the loop is:

`decode (async) -> global sync -> decode (async) -> global sync -> ...`

Within one decode step the splits are data-dependent and therefore serial
(GPU0 -> GPU1 -> GPU2 -> GPU3). The copies/events overlap never gets a chance to fire,
so only one GPU is busy at a time. This matches the ticket's observation.

Overlap only materialises when several `llama_decode` calls are issued before a
synchronize (speculative decoding, or a custom driver). Note the server skips the sync
when `has_output` is false, so consecutive prefill-only sub-batches can overlap.

Conclusion: the observed underutilisation is expected. The mechanism for overlap exists
but is not exercised by the stock server's generation loop.

---

## 3. Diagnostics decision

Decision: no code changes and no added logging are needed to characterise the baseline.
The following already suffice.

1. Startup log lines (from `llama-server`):
   - `graph ... splits = N` - number of scheduler splits (expect ~4-5 for 4 GPUs).
   - `reserve took ... sched copies = 4|1` - pipeline parallelism on/off.
   - `pipeline parallelism enabled` (present when on).
2. Per-GPU utilisation over time: `nvidia-smi dmon -s u -d 1`, or the existing
   `nvidia_gpu_exporter` + Grafana stack on rig1. The "one GPU busy at a time" signature
   is the primary evidence.
3. `llama-server /metrics` (enable `--metrics`): `llamacpp:n_decode_total`,
   `llamacpp:n_busy_slots_per_decode`, prompt/eval timings, and per-request timings in
   the `/completion` response.

Deferred (only if the baseline is ambiguous): Nsight Systems (`nsys`) per-GPU kernel
timelines. Adds Docker/Maxwell friction, so avoid unless needed.

---

## 4. Baseline benchmark design

### 4.1 Model

A dense ~30B quantised model (Q4_K_M) that (a) fits spread across 4x8 GB and (b) has
enough layers for a clean 4-way split (>= ~32 layers). Candidates already in the rig1
catalogue (see `~/llamacpp/models.ini`): NVIDIA-Nemotron-3.5-Lightning-30B, Qwen3.5-30B
class, gemma-4 26B. Final choice confirmed against `models.ini` on rig1.

### 4.2 Configuration

- Docker (everything runs in Docker; sudo forbidden).
- `llama-server -m <model> -sm layer -ngl all -ts 1,1,1,1 -ctk f16 -ctv f16 --metrics`
  plus `-c <ctx>` and `-np <slots>` sized for the concurrency under test.
- Maxwell has no tensor cores; flash-attn auto-resolves off. Confirm it does not error.
- Determinism for correctness: fixed `--seed`, greedy sampling (temperature 0 or
  `top-k 1`).

### 4.3 Workload

For N in {1, 4, 8, 16}: submit N concurrent requests, each a fixed prompt with a fixed
`n_predict` (e.g. 256 tokens), all launched together.

### 4.4 Metrics

- Aggregate throughput (tokens/s) and wall-clock time.
- Per-request latency.
- Per-GPU utilisation time-series (`nvidia-smi dmon`), plus avg/max per GPU.
- Correctness: greedy output identical to a single-device reference run (fixed seed).

### 4.5 Success criteria

- Baseline reproduces the "one GPU busy at a time" serial signature.
- `sched copies = 4` is confirmed in the log (pipeline parallelism is on, but the server
  loop serialises it).
- Throughput/latency recorded per concurrency level, correctness bit-identical.

---

## 5. Plan phases (tracked in issue #2)

- Phase 1: analysis of the parallelism mechanism (this document).
- Phase 2: diagnostics decision (section 3).
- Phase 3: baseline benchmark design (section 4).
- Phase 4 (complete): executed the baseline on rig1 with Qwen3.5-9B-Q4_K_M; recorded
  times, utilisation and correctness.
- Phase 5 (complete): wrote up the baseline results in
  `docs/experiments/pipeline-parallelism-baseline.md`.
