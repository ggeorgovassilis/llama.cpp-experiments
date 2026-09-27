# Pipeline parallelism baseline: results

Status: complete. Executed on rig1, 2026-09-27.

Tracked by: issue #2. Analysis and mechanism: `pipeline-parallelism-analysis.md`.

---

## 1. What was measured

The goal: quantify GPU utilisation and throughput when a layer-split model serves
parallel requests, and confirm whether the upstream `n_copies = 4` pipeline-parallelism
mechanism actually overlaps GPU work in the stock `llama-server`.

Method: greedy decoding (temperature 0, fixed seed) with a fixed prompt, N concurrent
`/completion` requests, for N in {1, 4, 8, 16}. Per-request and aggregate timing come
from the server's `/completion` timings block; per-GPU utilisation comes from
`nvidia-smi dmon -s u -d 1` sampled during each level.

Harness: `bench/` in this repo (`server.sh`, `client.py`, `summarize_dmon.py`,
`run_benchmark.sh`).

---

## 2. Model and configuration

- Model: `Qwen3.5-9B-Q4_K_M.gguf` (dense `qwen35`, 5.68 GB, 32 layers).
- Server: `llama-server -m ... -sm layer -ngl all -ts 1,1,1,1 -ctk f16 -ctv f16 -fa off
  -c 32768 -np 16 -ctxcp 0 --metrics`.
- 33/33 layers on GPU, no CPU offload. `pipeline parallelism enabled`, `graph splits = 5`,
  `sched copies = 4`.

Model selection: a dense model was required so the layer split is a clean 1:1 GPU
partition (no MoE experts). Qwen3.5-9B was chosen over the initial 27B and a 12B gemma
because its small `head_count_kv = 4` keeps the KV cache at ~256 MiB/GPU, leaving ~6 GB
headroom per GPU for parallelism. See the ticket for the full selection rationale.

---

## 3. Results

| N | wall (s) | tokens | aggregate tok/s | per-GPU SM (avg) | outputs identical |
|---|---|---|---|---|---|
| 1 | 19.9 | 64 | 3.22 | 21.6 - 31.2% | yes |
| 4 | 41.3 | 256 | 6.20 | 31.3 - 39.3% | yes |
| 8 | 79.5 | 512 | 6.44 | 35.7 - 39.6% | yes |
| 16 | 201.8 | 1024 | 5.07 | 31.5 - 36.2% | yes |

All levels: 0 request errors, all outputs non-empty and identical to each other.

---

## 4. Interpretation

Aggregate throughput rises from N=1 (3.22) to N=8 (6.44) then falls at N=16 (5.07).
Per-GPU SM averages stay flat at roughly 30-40% across every level (peaks hit 100%, but
only in short bursts).

The N=1 -> N=8 rise is **batching**, not pipeline overlap: more concurrent requests pack
more tokens per decode ubatch, filling otherwise-idle compute. The N=16 drop is slot
saturation - 16 concurrent requests contend for 16 slots and the queue/scheduler
overhead dominates.

The flat ~35% SM is the key finding. If the `n_copies = 4` overlap were firing, SM would
climb toward saturation as successive decodes overlapped. Instead it plateaus regardless
of concurrency - the "one GPU busy at a time" serial signature predicted in the analysis.
This confirms: the mechanism is on (`sched copies = 4`) but the stock server's
sync-after-every-decode loop serialises the layer pipeline, so overlap never fires during
generation.

---

## 5. Decisions recorded

- Iteration loop uses a single level (`LEVELS=8 PREDICT=64`, ~80 s). The full
  `1 4 8 16` sweep (this document) is the one-off baseline.
- `-ctxcp 0` disables context checkpoints; they snapshotted ~149 MiB of KV per slot and
  caused CUDA OOM at N=16 on larger models.
- gemma-4-12b abandoned: per-layer KV heads + SWA produced a ~2 GB KV buffer/GPU and
  OOM at N=16.

## 6. Open observations

- Output divergence (non-identical greedy outputs) was seen on Qwen3.8-27B and
  gemma-4-12b at N>=4, but not on Qwen3.5-9B. Divergence is model/arch specific, not a
  harness bug. Worth a follow-up (multi-GPU FP non-determinism vs prompt-cache/batch
  effects).
