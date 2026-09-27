# T9: KV cache quantisation is flat on sm_50

Tracks ticket #13. One-off run to record decode throughput and prefill across
KV cache types (f16 / q8_0 / q4_0) in tensor-split mode. Not part of the regular
benchmark loop.

## Background

All benchmark runs to date (#9, #10, #11) used f16 KV cache (`-ctk f16 -ctv f16`)
to match the layer-split baseline. The M10 has no tensor cores, so KV
quantisation was expected to trade cache precision for VRAM capacity rather than
speed. This run documents the actual numbers.

## Method

Driver `llama-pipeline` (this fork), `-sm tensor -fa on -ts 1,1,1,1`, K=1,
`-n 32`, prompt 128, `-c 1024`, f16 compute. `-ctk`/`-ctv` are the only varying
axis. 3 iterations each, average reported. Script:
`rig1/scripts/sweep_kv_quant.sh`.

## Results

### Qwen3.5-9B-Q4_K_M (dense, 5.68 GB)

| cache | prefill (ms) | decode (tok/s) |
| --- | ---: | ---: |
| f16 | 2059 | 15.81 |
| q8_0 | 2056 | 15.59 |
| q4_0 | 2068 | 15.45 |

### Qwen3.8-27B-UD-IQ4_XS (MoE, 14.25 GB)

| cache | prefill (ms) | decode (tok/s) |
| --- | ---: | ---: |
| f16 | 5465 | 6.22 |
| q8_0 | 5483 | 6.16 |
| q4_0 | 5461 | 6.15 |

## Conclusion

KV cache quantisation does not move decode throughput or prefill on sm_50: the
spread across cache types is within run-to-run noise (<2%). This matches the
expectation for a no-tensor-core part, where the cache read/write is not on the
compute critical path.

VRAM is likewise unmoved at this context length: at `-c 1024` the KV cache is
small relative to the weights (5.68 GB / 14.25 GB), so the capacity win of
q4_0 over f16 only materialises at much longer contexts or larger batches. That
capacity benefit is not exercised here and remains out of scope.

## Related

- #13 (this ticket), #10 (tensor split), #9 (decode ceiling).
- `docs/experiments/pipeline-parallelism-tensor-split.md` (the f16 baseline).
