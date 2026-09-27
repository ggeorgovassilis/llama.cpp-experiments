# T7: Tensor split lifts the decode ceiling

Tracks ticket #10. Resolves the output-head straggler found in #9 by switching to tensor split, and records the model/prompt sweep.

## Background

#9 pinned the 1.48x decode-overlap shortfall on the output head. Under `-sm layer` the vocab projection `output.weight` ([3584, 248320]) sits on the last GPU only (`dev_output = get_layer_buft_list(n_layer_all)` in `src/llama-model.cpp`), and `-ts` cannot move it. On GPU3 it is 132 matvecs at ~32 ms each, 36 % of GPU3's decode time, versus 0.54 ms for a layer matvec.

## Finding

No new code was needed. `-sm tensor` already shards the output head: `output.weight` carries `GGML_BACKEND_SPLIT_AXIS_1`, so the meta device splits the vocab axis 4 ways. The two proposed layer-mode fixes (split the vocab axis, replicate the full head) are redundant.

## Corrected assumptions

Three things assumed in the earlier write-ups are wrong:

- Flash attention DOES run on sm_50. `FLASH_ATTN_AVAILABLE` is on by default and `ggml_cuda_get_best_fattn_kernel` falls back to the generic TILE/VEC kernels when there are no tensor cores. The `-sm tensor` flash-attn gate is satisfiable on rig1.
- `llm_arch_supports_sm_tensor` returns true for the qwen3 family.
- The real `-sm tensor` constraint is the allreduce, not flash-attn. Comm init is nccl -> internal -> butterfly. NCCL is compiled in by default (`GGML_CUDA_NCCL` is ON and `libnccl-dev` ships in the build image), but `ncclCommInitAll` fails at runtime on sm_50 (`cudaMemPoolCreate` -> `cudaErrorNotSupported`, no CUDA VMM); the internal allreduce needs cc >= Volta AND exactly 2 devices (rig1 is sm_50 x 4). Tensor mode therefore lands on the host-mediated butterfly allreduce. Slow but functional, and not the bottleneck for the models tested. See `pipeline-parallelism-nccl.md` (issue #11).

## Method

Driver `llama-pipeline` (this fork, `examples/pipeline/pipeline.cpp`), K=1, `-n 32`, ctk/ctv f16. Prefill is init + decode + sync of the full prompt; decode wall/tok/s exclude prefill. Tensor = `-sm tensor -fa on`; layer = `-sm layer -fa off -ts 1,1,1,1`.

## Results

### Qwen3.5-9B-Q4_K_M (dense, 5.68 GB)

| prompt | layer prefill (s) | layer tok/s | tensor prefill (s) | tensor tok/s |
| --- | ---: | ---: | ---: | ---: |
| 128 | 4.45 | 5.32 | 2.11 | 15.76 |
| 512 | 8.41 | 5.59 | 3.69 | 15.83 |
| 1024 | 14.71 | 5.23 | 5.97 | 15.90 |
| 2048 | 21.45 | 5.03 | 10.85 | 16.02 |

Decode throughput is ~3x higher in tensor mode and flat versus prompt length in both modes. Prefill scales linearly with prompt length; tensor-mode prefill is ~2x faster.

### gemma-4-12b-it-Q4_K_M (dense, 7.12 GB)

| prompt | layer prefill (s) | layer tok/s | tensor |
| --- | ---: | ---: | --- |
| 128 | 6.69 | 3.78 | unsupported |
| 1024 | 23.51 | 3.27 | unsupported |

Tensor mode asserts in the meta backend (`ggml-backend-meta.cpp:543`, a `SPLIT_AXIS_0` tensor). Gemma4 is not tensor-split-compatible in this build. This is a genuine arch gap, not a rig1 constraint.

### Qwen3.8-27B-UD-IQ4_XS (MoE, 14.25 GB)

| prompt | layer prefill (s) | layer tok/s | tensor prefill (s) | tensor tok/s |
| --- | ---: | ---: | ---: | ---: |
| 128 | 16.64 | 1.99 | 5.48 | 6.22 |

Tensor mode works on the MoE and gives 3.1x decode and ~3x prefill.

## Notes

- Prompt 2048 needs `-b 4096`: the driver feeds the whole prompt as one batch and the default `n_batch` is 2048, so it asserts `n_tokens_all <= n_batch`.
- gemma-4 tensor mode is an arch gap (separate ticket if gemma4 tensor-split matters).

## Conclusion

Sharding the output head is the right lever, and it is already implemented as `-sm tensor`. Tensor mode gives ~3x decode on the 9B dense and the 27B MoE, and ~2-3x faster prefill. This lifts the decode ceiling from the ~1.5x measured in #9 toward ~3x.

## Open questions

- Does `-sm tensor` scale with K (2, 4), or does the shared meta device serialise the contexts? Answered in #11: flat over K (15.83 -> 16.58 tok/s, +4.7%) because tensor parallelism already uses all 4 GPUs per token. See `pipeline-parallelism-nccl.md`.
- Is NCCL worth enabling (bigger models / higher allreduce volume)? Answered in #11: no - NCCL cannot initialise on sm_50 (no CUDA VMM). See `pipeline-parallelism-nccl.md`.
- Is gemma4 tensor-split support worth a follow-up?

## Related

- #9 (decode ceiling, where the straggler was found).
- #10 (this ticket).
- `docs/experiments/pipeline-parallelism-decode-ceiling.md` (the straggler analysis).
- `docs/multi-gpu.md` (split modes and flags).
