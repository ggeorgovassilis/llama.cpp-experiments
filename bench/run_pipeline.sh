#!/usr/bin/env bash
# Run the pipeline-parallelism PoC driver (issue #4) on rig1.
#
# For each concurrency K in {1,2,4} and each mode (overlap, serial) it runs the
# llama-pipeline driver layer-split across the 4 GPUs, samples per-GPU
# utilisation (nvidia-smi dmon) during the run, and records the driver output.
#
# Usage: ./run_pipeline.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HOME/llamacpp-experiments/bench-results/pipeline"
BUILD="$HOME/llamacpp-experiments/build"
MODELS="/mnt/ssd2/models"
IMAGE="llamacpp-exp-build:12.6.2"
MODEL="${MODEL:-Qwen3.5-9B-Q4_K_M.gguf}"
PREDICT="${PREDICT:-64}"
PROMPT_LEN="${PROMPT_LEN:-128}"
CTX="${CTX:-1024}"
LEVELS="${LEVELS:-1 2 4}"

mkdir -p "$OUT"

run_one() {
    local k="$1"
    local mode="$2"
    local extra=""
    if [[ "$mode" == "serial" ]]; then
        extra="--serial"
    fi
    local tag="k${k}_${mode}"
    echo "=== K=$k mode=$mode ==="

    nvidia-smi dmon -s u -d 1 > "$OUT/gpu_${tag}.log" 2>&1 &
    local sampler=$!

    docker run --rm --gpus all \
        -v "$BUILD:/build" \
        -v "$MODELS:/models:ro" \
        "$IMAGE" /build/bin/llama-pipeline \
            -m "/models/$MODEL" \
            -sm layer -ngl all -ts 1,1,1,1 \
            -ctk f16 -ctv f16 -fa off \
            -c "$CTX" -n "$PREDICT" --prompt-len "$PROMPT_LEN" \
            -k "$k" $extra \
            > "$OUT/run_${tag}.log" 2>&1

    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true

    echo "--- driver summary ---"
    grep -E "RESULT|contexts|aggregate_tok_s" "$OUT/run_${tag}.log" || true
    echo "--- GPU utilisation ---"
    python3 "$HERE/summarize_dmon.py" "$OUT/gpu_${tag}.log"
}

for k in $LEVELS; do
    run_one "$k" overlap
    run_one "$k" serial
done

echo "=== done. results in $OUT ==="
