#!/usr/bin/env bash
# Start / stop the llama-server used by the pipeline-parallelism baseline benchmark.
#
# The container runs detached (no -t / -i) so it never falls into the
# "spamming >" interactive-prompt loop.
#
# Usage: ./server.sh start|stop|logs
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BUILD="$HOME/llamacpp-experiments/build"
MODELS="/mnt/ssd2/models"
IMAGE="llamacpp-exp-build:12.6.2"

SERVER_NAME="pp-bench-server"
PORT="${PORT:-18080}"
MODEL="${MODEL:-Qwen3.5-9B-Q4_K_M.gguf}"
CTX="${CTX:-32768}"
NP="${NP:-16}"
LOG="$HOME/llamacpp-experiments/bench-results/server.log"

mkdir -p "$(dirname "$LOG")"

start() {
    docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
    docker run -d --name "$SERVER_NAME" --gpus all \
        -v "$BUILD:/build" \
        -v "$MODELS:/models:ro" \
        -p "$PORT:8080" \
        "$IMAGE" /build/bin/llama-server \
            -m "/models/$MODEL" \
            -sm layer -ngl all -ts 1,1,1,1 \
            -ctk f16 -ctv f16 -fa off \
            -c "$CTX" -np "$NP" \
            -ctxcp 0 -lv 4 \
            --metrics --host 0.0.0.0 --port 8080

    echo "waiting for server health on 127.0.0.1:$PORT ..."
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "healthy"
            return 0
        fi
        if ! docker inspect -f '{{.State.Running}}' "$SERVER_NAME" 2>/dev/null | grep -q true; then
            echo "container exited early; last log lines:" >&2
            docker logs --tail 60 "$SERVER_NAME" >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "timed out waiting for health" >&2
    docker logs --tail 60 "$SERVER_NAME" >&2 || true
    return 1
}

stop() {
    docker logs --tail 200 "$SERVER_NAME" > "$LOG" 2>/dev/null || true
    docker rm -f "$SERVER_NAME" >/dev/null 2>&1 || true
}

logs() {
    docker logs "$SERVER_NAME" 2>&1 || true
}

case "${1:-}" in
    start) start ;;
    stop)  stop ;;
    logs)  logs ;;
    *) echo "usage: $0 start|stop|logs" >&2; exit 2 ;;
esac
