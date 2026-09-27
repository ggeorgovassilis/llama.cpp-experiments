#!/usr/bin/env bash
# Build llama.cpp for rig1.
#   - GPU: 4x Tesla M10 = Maxwell sm_50  -> -DCMAKE_CUDA_ARCHITECTURES=50
#   - CPU: 2x Xeon E5-2640 = Sandy Bridge -> AVX only (no AVX2/FMA/F16C)
# Compiling for a single CUDA arch is what makes the build "bespoke and fast":
# it skips the default multi-arch (50/61/70/75/80/86/89/90/120) matrix.
#
# Usage: ./scripts/build.sh [clean]
#   clean  - wipe the build dir and ccache before building
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SRC="$ROOT/src"
BUILD="$ROOT/build"
CCACHE="$ROOT/.ccache"
IMAGE="llamacpp-exp-build:12.6.2"

mkdir -p "$BUILD" "$CCACHE"

if [[ "${1:-}" == "clean" ]]; then
    rm -rf "$BUILD" "$CCACHE"
    mkdir -p "$BUILD" "$CCACHE"
fi

docker build -t "$IMAGE" -f "$ROOT/docker/Dockerfile.build" "$ROOT/docker"

exec docker run --rm --gpus all \
    -v "$SRC:/app:ro" \
    -v "$BUILD:/build" \
    -v "$CCACHE:/ccache" \
    -e CCACHE_DIR=/ccache \
    "$IMAGE" \
    bash -c '
set -euo pipefail
cmake -S /app -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=50 \
    -DGGML_NATIVE=OFF \
    -DGGML_SSE42=ON \
    -DGGML_AVX=ON \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_CUB_3DOT2=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build /build
'
