#!/usr/bin/env bash
# Run the pipeline-parallelism baseline benchmark on rig1.
#
# For each concurrency level N in {1,4,8,16} it:
#   1. starts llama-server (layer-split across 4 GPUs) if not already running
#   2. warms up with a single request
#   3. samples per-GPU utilisation (nvidia-smi dmon) while the client submits N
#      concurrent requests
#   4. writes results and the dmon log
#
# Usage: ./run_benchmark.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/llamacpp-experiments/bench-results"
URL="http://127.0.0.1:${PORT:-18080}"

mkdir -p "$OUT"

levels="${LEVELS:-1 4 8 16}"
PREDICT="${PREDICT:-64}"
PROMPT_LEN="${PROMPT_LEN:-128}"

for N in $levels; do
    echo "=== N=$N ==="

    # warm up once per level so the model/KV/graphs are resident
    python3 "$HERE/client.py" --url "$URL" --n 1 --predict 16 \
        --out "$OUT/warmup_n${N}.json" >/dev/null

    # sample per-GPU utilisation during the run
    nvidia-smi dmon -s u -d 1 > "$OUT/gpu_n${N}.log" 2>&1 &
    SAMPLER=$!

    python3 "$HERE/client.py" --url "$URL" --n "$N" --predict "$PREDICT" \
        --prompt-len "$PROMPT_LEN" --out "$OUT/results_n${N}.json"

    kill "$SAMPLER" 2>/dev/null || true
    wait "$SAMPLER" 2>/dev/null || true

    echo "--- GPU utilisation N=$N ---"
    python3 "$HERE/summarize_dmon.py" "$OUT/gpu_n${N}.log"
done

echo "=== done. results in $OUT ==="
