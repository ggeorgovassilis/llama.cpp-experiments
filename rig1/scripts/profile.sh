#!/usr/bin/env bash
# Profile the llama-pipeline driver with Nsight Systems to capture per-GPU
# kernel timelines of the decode loop on rig1.
#
# Usage (from ~/llamacpp-experiments):
#   ./scripts/profile.sh                 # build image + capture nsys trace
#   ./scripts/profile.sh ncu             # run Nsight Compute on one kernel
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="llamacpp-exp-profile:12.6.2"
MODEL="/models/Qwen3.5-9B-Q4_K_M.gguf"
RESULTS="$ROOT/profile-results"
mkdir -p "$RESULTS"

docker build -t "$IMAGE" -f "$ROOT/docker/Dockerfile.profile" "$ROOT/docker"

COMMON=(
    -m "$MODEL"
    -sm layer -ngl all -ts 1,1,1,1 -ctk f16 -ctv f16 -fa off
    -c 1024 -n 32 --prompt-len 128 -k 4
)

if [[ "${1:-}" == "ncu" ]]; then
    exec docker run --rm --gpus all --privileged \
        -v "$ROOT/build:/build" \
        -v /mnt/ssd2/models:/models:ro \
        -v "$RESULTS:/results" \
        "$IMAGE" \
        ncu --set full --launch-count 1 --launch-skip 200 \
            -o /results/ncu \
            /build/bin/llama-pipeline "${COMMON[@]}"
fi

# default: nsys CUDA trace of the decode loop
exec docker run --rm --gpus all --privileged \
    -v "$ROOT/build:/build" \
    -v /mnt/ssd2/models:/models:ro \
    -v "$RESULTS:/results" \
    "$IMAGE" \
    nsys profile --trace=cuda,osrt --force-overwrite=true \
        -o /results/pipeline_overlap \
        /build/bin/llama-pipeline "${COMMON[@]}"
