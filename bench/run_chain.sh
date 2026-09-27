#!/usr/bin/env bash
# Run the chain-decode PoC driver (issue #5, Theory B) on rig1.
#
# Single context, single sequence: M decodes issued back-to-back before one sync.
# Sweeps M in {2,4,8} and toggles LLAMA_GRAPH_REUSE_DISABLE (0 = default graph
# reuse, 1 = force copy rotation / overlap). Captures per-GPU SM via dmon.
#
# Usage: ./run_chain.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/llamacpp-experiments/bench-results/chain"
BUILD="$HOME/llamacpp-experiments/build"
MODELS="/mnt/ssd2/models"
IMAGE="llamacpp-exp-build:12.6.2"
MODEL="${MODEL:-Qwen3.5-9B-Q4_K_M.gguf}"
PROMPT_LEN="${PROMPT_LEN:-128}"
CTX="${CTX:-1024}"
REPEATS="${REPEATS:-16}"
LEVELS="${LEVELS:-2 4 8}"

mkdir -p "$OUT"

run_one() {
    local m="$1"
    local reuse="$2"
    local tag="m${m}_reuse${reuse}"
    echo "=== M=$m reuse=$reuse ==="

    nvidia-smi dmon -s u -d 1 > "$OUT/gpu_${tag}.log" 2>&1 &
    local sampler=$!

    docker run --rm --gpus all \
        -e "LLAMA_GRAPH_REUSE_DISABLE=${reuse}" \
        -v "$BUILD:/build" \
        -v "$MODELS:/models:ro" \
        "$IMAGE" /build/bin/llama-chain \
            -m "/models/$MODEL" \
            -sm layer -ngl all -ts 1,1,1,1 \
            -ctk f16 -ctv f16 -fa off \
            -c "$CTX" --prompt-len "$PROMPT_LEN" \
            --chain "$m" --repeats "$REPEATS" \
            > "$OUT/run_${tag}.log" 2>&1

    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true

    echo "--- driver summary ---"
    grep -E "RESULT|graph reuse disabled" "$OUT/run_${tag}.log" || true
    echo "--- GPU utilisation ---"
    python3 "$HERE/summarize_dmon.py" "$OUT/gpu_${tag}.log"
}

for m in $LEVELS; do
    run_one "$m" 0
    run_one "$m" 1
done

echo "=== done. results in $OUT ==="
