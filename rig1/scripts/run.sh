#!/usr/bin/env bash
# Smoke test: load a small model onto the GPU and generate a few tokens.
# Proves the sm_50 CUDA build actually runs inference end-to-end on the M10.
#
# Usage: ./scripts/run.sh [model.gguf] [n-tokens]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD="$ROOT/build"
IMAGE="llamacpp-exp-build:12.6.2"
MODELS_DIR="/mnt/ssd2/models"

MODEL="${1:-Qwen3.5-0.8B-Q8_0.gguf}"
NTOK="${2:-16}"

exec docker run --rm --gpus all \
    -v "$BUILD:/build" \
    -v "$MODELS_DIR:/models:ro" \
    -e CCACHE_DIR=/ccache \
    "$IMAGE" \
    /build/bin/llama-cli \
        -m "/models/${MODEL}" \
        -p "The capital of France is" \
        -n "$NTOK" \
        -ngl 99 \
        --single-turn \
        --no-display-prompt
