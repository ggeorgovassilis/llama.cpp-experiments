#!/usr/bin/env bash
# Prefill baseline for the layer-split model (issue #8): sweep prompt length and
# compare graph-reuse ON vs OFF (LLAMA_GRAPH_REUSE_DISABLE) to test whether
# consecutive prefill ubatches overlap.
#
# The driver feeds the whole prompt as one batch; llama_decode splits it into
# n_ubatch (512) sub-batches internally. Measured result: reuse on/off is a no-op
# for prefill, because the KQ mask grows every sub-batch so graph reuse never
# fires (see docs/experiments/prefill-parallelism-baseline.md).
#
# Usage (from ~/llamacpp-experiments): ./scripts/sweep_prefill.sh [iterations]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
IMAGE="llamacpp-exp-build:12.6.2"
MODEL="/models/Qwen3.5-9B-Q4_K_M.gguf"
ITER="${1:-2}"
PROMPTS="${PROMPTS:-512 1024 2048 4096}"

run_one() {
  local prompt_len="$1"
  local reuse="$2"   # "on" or "off"
  local extra=()
  if [[ "$reuse" == "off" ]]; then
    extra=(-e LLAMA_GRAPH_REUSE_DISABLE=1)
  fi
  docker run --rm --gpus all \
    -v "$ROOT/build:/build" \
    -v /mnt/ssd2/models:/models:ro \
    "${extra[@]}" \
    "$IMAGE" \
    /build/bin/llama-pipeline \
      -m "$MODEL" -sm layer -ngl all -ts 1,1,1,1 -ctk f16 -ctv f16 -fa off \
      -c 8192 -b 8192 -n 1 --prompt-len "$prompt_len" -k 1 2>&1 \
    | grep -oE 'prompt_tokens=[0-9]+|prefill_ms=[0-9.]+' | tr '\n' ' '
}

echo "prefill sweep (model=Qwen3.5-9B-Q4_K_M, layer split, K=1, n=1, iterations=$ITER)"
for pl in $PROMPTS; do
  for reuse in on off; do
    best_tok_s=0
    for _ in $(seq 1 "$ITER"); do
      out=$(run_one "$pl" "$reuse")
      pt=$(echo "$out" | grep -oE 'prompt_tokens=[0-9]+' | head -1 | cut -d= -f2)
      pm=$(echo "$out" | grep -oE 'prefill_ms=[0-9.]+' | head -1 | cut -d= -f2)
      if [[ -z "$pt" || -z "$pm" ]]; then
        echo "  prompt=$pl reuse=$reuse  ERROR: no timing (pt='$pt' pm='$pm')" >&2
        continue
      fi
      tok_s=$(awk -v p="$pt" -v m="$pm" 'BEGIN{printf "%.2f", p*1000/m}')
      best_tok_s=$(awk -v a="$best_tok_s" -v b="$tok_s" 'BEGIN{print (b>a)?b:a}')
    done
    printf "  prompt=%-5d reuse=%-3s  best=%-8s tok/s (prefill, n_tokens=%s)\n" "$pl" "$reuse" "$best_tok_s" "${pt:-?}"
  done
done
