#!/usr/bin/env bash
# Scale K (number of independent contexts) in overlap mode and report aggregate
# decode tok/s. Shows whether the overlap ceiling is compute-bound (linear) or
# sync-bound (sub-linear/plateau).
#
# Usage (from ~/llamacpp-experiments): ./scripts/sweep_k.sh [iterations]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="llamacpp-exp-build:12.6.2"
MODEL="/models/Qwen3.5-9B-Q4_K_M.gguf"
ITER="${1:-3}"

run_one() {
  local k="$1"
  docker run --rm --gpus all \
    -v "$ROOT/build:/build" \
    -v /mnt/ssd2/models:/models:ro \
    "$IMAGE" \
    /build/bin/llama-pipeline \
      -m "$MODEL" -sm layer -ngl all -ts 1,1,1,1 -ctk f16 -ctv f16 -fa off \
      -c 1024 -n 32 --prompt-len 128 -k "$k" 2>/dev/null \
    | grep -oE 'tok_s=[0-9.]+' | head -1
}

echo "overlap scaling over K (iterations=$ITER)"
for k in 1 2 3 4 6 8; do
  total=0
  best=0
  for _ in $(seq 1 "$ITER"); do
    v=$(run_one "$k")
    v=${v#tok_s=}
    total=$(awk -v a="$total" -v b="$v" 'BEGIN{print a+b}')
    best=$(awk -v a="$best" -v b="$v" 'BEGIN{print (b>a)?b:a}')
  done
  avg=$(awk -v t="$total" -v n="$ITER" 'BEGIN{printf "%.2f", t/n}')
  printf "  K=%-2d  avg=%-6s tok/s   best=%-6s tok/s\n" "$k" "$avg" "$best"
done
