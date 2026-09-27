# true-pipeline branch: user-facing changes vs upstream

Consolidated list of what the `true-pipeline` branch adds over upstream `master`.
Pointers into the per-experiment write-ups under `docs/experiments/`.

## Drivers

### llama-pipeline - parallel independent contexts (Theory A, #4)

Driver at `examples/pipeline/pipeline.cpp` (binary `llama-pipeline`). Loads a
dense model layer-split across all GPUs, creates K independent contexts with the
same prompt, and decodes greedily. Default "overlap" mode issues the K decodes
back-to-back before synchronising, so the per-GPU layer pipeline of one context
overlaps the still-running pipeline of another; `--serial` decodes and syncs one
context at a time (matching the stock server loop).

- Flags: `-k/--n-contexts`, `--serial`, `--prompt-len`
- Result: +47.9% at K=4
- Write-up: `docs/experiments/pipeline-parallelism-overlap.md`

### llama-chain - speculative/draft decode overlap (Theory B, #5)

Driver at `examples/pipeline/chain.cpp` (binary `llama-chain`). Single context,
single sequence: M decodes issued back-to-back before one sync, compared against
the serial reference (decode -> sync per step).

- Flags: `--chain M`, `--repeats`, `--prompt-len`
- Env: `LLAMA_GRAPH_REUSE_DISABLE` (0/1) toggles the graph-reuse fast path
- Result: 2.15x at M=8 with graph reuse disabled
- Write-up: `docs/experiments/pipeline-parallelism-chain.md`

## Env knobs (code level)

- `LLAMA_PIPELINE_PARALLEL` (`src/llama-context.cpp`): `=0` forces
  `pipeline_parallel` off and `n_copies = 1` (#7). Write-up:
  `docs/experiments/pipeline-parallelism-control.md`.
- `LLAMA_GRAPH_REUSE_DISABLE` (`src/llama-context.cpp`): `=1` forces graph
  re-allocation per decode instead of the reuse fast path.

## Bench harness

`bench/` holds the reproduction tooling:

- `run_pipeline.sh` - sweeps K and mode for `llama-pipeline`, captures per-GPU
  utilisation with `nvidia-smi dmon`
- `run_chain.sh` - sweeps M and the graph-reuse toggle for `llama-chain`
- `server.sh` - server driver
- `summarize_dmon.py` - aggregates the dmon captures
- `client.py`, `gguf_info.py`, `run_benchmark.sh` - load/request helpers

## Findings

- Decode ceiling (#9): the output head is the straggler; cross-GPU transfer is
  idle. Write-up: `docs/experiments/pipeline-parallelism-decode-ceiling.md`.
- Tensor split (#10): `-sm tensor` shards the output head, ~3x decode. Note the
  gemma4 tensor-split arch gap and the NCCL/butterfly allreduce caveat (tracked
  separately in #11). Write-up:
  `docs/experiments/pipeline-parallelism-tensor-split.md`.
