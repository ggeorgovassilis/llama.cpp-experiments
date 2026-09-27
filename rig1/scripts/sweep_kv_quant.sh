#!/usr/bin/env bash
# One-off KV cache quantisation sweep (issue #13). Runs -sm tensor at K=1 and
# reports prefill time and decode tok/s for each cache type (f16 / q8_0 / q4_0).
# This is a one-time documentation run, not part of the regular benchmark loop.
#
# Usage (from ~/llamacpp-experiments): ./scripts/sweep_kv_quant.sh [iterations]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="llamacpp-exp-build:12.6.2"
MODELS=(
  "/models/Qwen3.5-9B-Q4_K_M.gguf"
  "/models/Qwen3.8-27B-UD-IQ4_XS.gguf"
)
CACHE_TYPES=(f16 q8_0 q4_0)
ITER="${1:-3}"

run_one() {
  local model="$1" ctk="$2" ctv="$3"
  # prefill_ms is printed to stderr, tok_s to stdout; merge both.
  docker run --rm --gpus all \
    -v "$ROOT/build:/build" \
    -v /mnt/ssd2/models:/models:ro \
    "$IMAGE" \
    /build/bin/llama-pipeline \
      -m "$model" -sm tensor -ngl all -ts 1,1,1,1 -ctk "$ctk" -ctv "$ctv" -fa on \
      -c 1024 -n 32 --prompt-len 128 -k 1 2>&1 \
    | grep -oE 'prefill_ms=[0-9.]+|tok_s=[0-9.]+' | tr '\n' ' '
}

echo "KV quant sweep (tensor mode, K=1, iterations=$ITER)"
for model in "${MODELS[@]}"; do
  echo "== $(basename "$model") =="
  for ctk in "${CACHE_TYPES[@]}"; do
    p_total=0
    t_total=0
    t_best=0
    for _ in $(seq 1 "$ITER"); do
      out=$(run_one "$model" "$ctk" "$ctk")
      p=$(echo "$out" | grep -oE 'prefill_ms=[0-9.]+' | head -1 | cut -d= -f2)
      t=$(echo "$out" | grep -oE 'tok_s=[0-9.]+' | head -1 | cut -d= -f2)
      p=${p:-0}
      t=${t:-0}
      p_total=$(awk -v a="$p_total" -v b="$p" 'BEGIN{print a+b}')
      t_total=$(awk -v a="$t_total" -v b="$t" 'BEGIN{print a+b}')
      t_best=$(awk -v a="$t_best" -v b="$t" 'BEGIN{print (b>a)?b:a}')
    done
    p_avg=$(awk -v t="$p_total" -v n="$ITER" 'BEGIN{printf "%.0f", t/n}')
    t_avg=$(awk -v t="$t_total" -v n="$ITER" 'BEGIN{printf "%.2f", t/n}')
    printf "  ctk/ctv=%-6s  prefill_avg=%-6s ms   tok_s avg=%-6s  best=%-6s\n" \
      "$ctk" "$p_avg" "$t_avg" "$t_best"
  done
done
