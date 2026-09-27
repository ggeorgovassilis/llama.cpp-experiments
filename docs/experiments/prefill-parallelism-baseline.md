# Prefill parallelism baseline: results

Status: complete. Executed on rig1, 2026-09-27.

Tracked by: issue #8. Mechanism and (corrected) analysis:
`prefill-parallelism-analysis.md`.

---

## 1. What was measured

The goal: quantify prefill throughput for a layer-split model and test whether
disabling graph reuse (`LLAMA_GRAPH_REUSE_DISABLE=1`) unlocks more overlap between
consecutive prefill sub-batches.

Method: a single context (`K=1`), one prefill of a fixed prompt, no generation
(`n_predict=1` so decode is negligible). The driver `examples/pipeline/llama-pipeline`
reports `prefill_ms` around the prompt decode. Prompt lengths {512, 1024, 2048, 4096},
each run twice (reuse on, reuse off), two iterations each, best prefill tok/s kept.
Pre-fill tok/s = `prompt_tokens * 1000 / prefill_ms`.

Harness: `rig1/scripts/sweep_prefill.sh`.

---

## 2. Model and configuration

- Model: `Qwen3.5-9B-Q4_K_M.gguf` (dense `qwen35`, 5.68 GB, 32 layers).
- Driver: `llama-pipeline -m ... -sm layer -ngl all -ts 1,1,1,1 -ctk f16 -ctv f16
  -fa off -c 8192 -b 8192 -n 1 --prompt-len <N> -k 1`.
- 33/33 layers on GPU, no CPU offload. `pipeline parallelism enabled`, `sched copies = 4`
  (confirmed with `--verbose`; these INFO lines are hidden at the default log level).

Model selection follows issue #2: a dense model so the layer split is a clean 1:1 GPU
partition (no MoE experts). Qwen3.5-9B is the validated baseline model from #2/#10.

---

## 3. Results

| prompt len | tokens | reuse on (tok/s) | reuse off (tok/s) |
|---|---|---|---|
| 512 | 534 | 63.07 | 62.88 |
| 1024 | 1026 | 69.73 | 69.63 |
| 2048 | 2051 | 95.46 | 95.29 |
| 4096 | 4101 | 125.48 | 125.23 |
| 8192 | 8201 | 153.97 | - |

Reuse on vs off differ by less than 0.3% at every length - noise. Disabling graph
reuse has no measurable effect on prefill throughput.

The 8192 row is a single run taken during the GPU-utilisation sample (section 4), not
part of the sweep.

---

## 4. GPU utilisation during a long prefill

`nvidia-smi` sampled every 3 s during a single 8192-token prefill (~53 s):

- t=12s: GPU0 100%, others idle (first sub-batch, GPU0 first).
- t=18s: GPU0 100%, GPU3 100%.
- t=21s: GPU0 100%, GPU1 100%.
- t=24s: GPU2 100%, GPU3 100%.
- t=36s: GPU0/GPU1/GPU2 100%.
- t=39s: all four at 100% (90/74 on the middle two).
- t=51s: all four at 100%.

All four GPUs are busy simultaneously for sustained periods. This is the signature of
cross-sub-batch pipeline overlap, not the "one GPU at a time" serial pattern. The
oscillation (GPUs cycling 0% -> 100%) is the sub-batch wave, with each GPU bursting
through its layer slice and idling briefly at the boundary.

Memory at steady state: GPU0 2266 MiB, GPU1 2116, GPU2 2113, GPU3 2809 MiB. The layer
split is not balanced - GPU3 carries more (later layers + output head).

---

## 5. Interpretation

Prefill is already parallelised across the 4 GPUs. Consecutive prefill sub-batches
overlap via the `cur_copy`/`next_copy` mechanism, so the answer to issue #8's "can
prefill be parallelised" is: it already is, and there is no graph-reuse barrier to
remove.

The reuse toggle is a no-op because graph reuse never fires during prefill. The
attention KQ mask (`kq_mask->ne[0] == n_kv`) grows with every sub-batch (512 -> 1024 ->
1536 -> ...), so `can_reuse` is false and every sub-batch is freshly allocated. This
contrasts with decode, where `n_kv` is padded to 256-token blocks and stays constant
for many consecutive steps, so decode does reuse (and pays the reuse sync). Full
explanation: `prefill-parallelism-analysis.md` section 3.2.

The 2.4x throughput rise from 512 to 8192 tokens is pipeline overlap plus batch
efficiency, but it does not reach the ~4x a perfectly balanced, fully overlapped split
would give. The gap is layer imbalance (GPU3 is ~35% heavier than GPU1/GPU2) and
pipeline bubbles at the start/end of each prefill - a separate question from the reuse
barrier issue #8 set out to test.

---

## 6. Decisions recorded

- `n_predict=1` keeps decode negligible, so `prefill_ms` is the whole story. The
  reuse-off run is therefore a diagnostic for prefill only (reuse-off would hurt decode,
  which depends on graph reuse - see `prefill-parallelism-analysis.md` section 3.2).
- The driver reports `prefill_ms` at the default log level; the
  `pipeline parallelism enabled` / `sched copies = 4` lines need `--verbose`.
