#!/usr/bin/env bash
# Build the runtime image that bundles the fork's binaries as a standalone
# deployment artifact. See docs/experiments/runtime-image.md.
#
# Usage: ./scripts/build-runtime.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD="$ROOT/build"
CTX="$ROOT/docker"
STAGE="$CTX/build/bin"
IMAGE="llamacpp-exp:true-pipeline"
VTAG="llamacpp-exp:12.6.2-true-pipeline"

# The runtime image COPYs build/bin, so the binaries must exist first.
if [[ ! -x "$BUILD/bin/llama-server" ]]; then
    echo "==> binaries missing, running build.sh first"
    "$HERE/build.sh"
fi

# Stage the four binaries plus their shared libs into the docker context dir.
rm -rf "$STAGE"
mkdir -p "$STAGE"
for b in llama-server llama-cli llama-pipeline llama-chain; do
    cp -a "$BUILD/bin/$b" "$STAGE/"
done
cp -a "$BUILD/bin"/lib*.so* "$STAGE/"

docker build -t "$IMAGE" -t "$VTAG" -f "$CTX/Dockerfile.runtime" "$CTX"

echo "==> built $IMAGE ($VTAG)"
