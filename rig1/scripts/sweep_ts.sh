#!/usr/bin/env bash
# Sweep -ts (tensor split) ratios for the layer-split decode pipeline and
# report aggregate decode tok/s for each. Uses the non-profiled build image
# for clean timing.
#
# Usage (from ~/llamacpp-experiments): ./scripts/sweep_ts.sh [iterations]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="llamacpp-exp-build:12.6.2"
MODEL="/models/Qwen3.5-9B-Q4_K_M.gguf"
ITER="${1:-3}"

# candidate splits -> (GPU0,GPU1,GPU2,GPU3) layer counts, output head on GPU3
TS_VALUES=(
  "1,1,1,1"     # 9,8,8,7 + output (baseline)
  "9,9,9,6"     # 9,9,9,5 + output
  "9,9,10,5"    # 9,9,10,4 + output (predicted optimum)
  "9,10,10,4"   # 9,10,10,3 + output
  "10,10,10,3"  # 10,10,10,2 + output
  "10,10,11,2"  # 10,10,11,1 + output
)

run_one() {
  local ts="$1"
  docker run --rm --gpus all \
    -v "$ROOT/build:/build" \
    -v /mnt/ssd2/models:/models:ro \
    "$IMAGE" \
    /build/bin/llama-pipeline \
      -m "$MODEL" -sm layer -ngl all -ts "$ts" -ctk f16 -ctv f16 -fa off \
      -c 1024 -n 32 --prompt-len 128 -k 4 2>/dev/null \
    | grep -oE 'tok_s=[0-9.]+' | head -1
}

echo "sweep over -ts (iterations=$ITER)"
for ts in "${TS_VALUES[@]}"; do
  total=0
  best=0
  for _ in $(seq 1 "$ITER"); do
    v=$(run_one "$ts")
    v=${v#tok_s=}
    total=$(awk -v a="$total" -v b="$v" 'BEGIN{print a+b}')
    best=$(awk -v a="$best" -v b="$v" 'BEGIN{print (b>a)?b:a}')
  done
  avg=$(awk -v t="$total" -v n="$ITER" 'BEGIN{printf "%.2f", t/n}')
  printf "  -ts %-10s  avg=%-6s tok/s   best=%-6s tok/s\n" "$ts" "$avg" "$best"
done
