#!/usr/bin/env bash
# Sweep -ts (tensor split) ratios for the layer-split PREfill pipeline and
# report prefill tok/s for each. Companion to sweep_ts.sh, which measured the
# same ratios for decode (issue #8 follow-up: does rebalancing layers away
# from GPU3 help prefill, where the output head is a smaller relative share).
#
# Usage (from ~/llamacpp-experiments): ./scripts/sweep_prefill_ts.sh [iterations]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="llamacpp-exp-build:12.6.2"
MODEL="/models/Qwen3.5-9B-Q4_K_M.gguf"
ITER="${1:-2}"
PROMPT_LEN="${PROMPT_LEN:-2048}"

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
      -c 8192 -b 8192 -n 1 --prompt-len "$PROMPT_LEN" -k 1 2>&1 \
    | grep -oE 'prompt_tokens=[0-9]+|prefill_ms=[0-9.]+' | tr '\n' ' '
}

echo "prefill -ts sweep (model=Qwen3.5-9B-Q4_K_M, layer split, K=1, prompt=$PROMPT_LEN, iterations=$ITER)"
for ts in "${TS_VALUES[@]}"; do
  total=0
  best=0
  n_tokens="?"
  for _ in $(seq 1 "$ITER"); do
    out=$(run_one "$ts")
    pt=$(echo "$out" | grep -oE 'prompt_tokens=[0-9]+' | head -1 | cut -d= -f2)
    pm=$(echo "$out" | grep -oE 'prefill_ms=[0-9.]+' | head -1 | cut -d= -f2)
    if [[ -z "$pt" || -z "$pm" ]]; then
      echo "  -ts $ts  ERROR: no timing (pt='$pt' pm='$pm')" >&2
      continue
    fi
    n_tokens="$pt"
    tok_s=$(awk -v p="$pt" -v m="$pm" 'BEGIN{printf "%.2f", p*1000/m}')
    total=$(awk -v a="$total" -v b="$tok_s" 'BEGIN{print a+b}')
    best=$(awk -v a="$best" -v b="$tok_s" 'BEGIN{print (b>a)?b:a}')
  done
  avg=$(awk -v t="$total" -v n="$ITER" 'BEGIN{printf "%.2f", t/n}')
  printf "  -ts %-10s  avg=%-7s tok/s   best=%-7s tok/s   (prefill, n_tokens=%s)\n" "$ts" "$avg" "$best" "$n_tokens"
done
