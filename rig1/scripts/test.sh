#!/usr/bin/env bash
# Run llama.cpp tests on rig1 inside the build container.
# Runs the fast unit tests plus the CUDA backend-op suite (test-backend-ops)
# to validate the sm_50 CUDA kernels against the CPU reference.
#
# Usage: ./scripts/test.sh [ctest-regex]
#   Default regex runs unit tests + backend-ops.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD="$ROOT/build"
IMAGE="llamacpp-exp-build:12.6.2"

REGEX="${1:-test-(alloc|argsort|double-float|grad0|opt|quantize-fns|rope|sampling|backend-ops)}"

exec docker run --rm --gpus all \
    -v "$BUILD:/build" \
    -e CCACHE_DIR=/ccache \
    "$IMAGE" \
    bash -c "
set -euo pipefail
cd /build
ctest --output-on-failure -R '${REGEX}'
"
