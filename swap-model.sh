#!/bin/bash
set -e

LLAMA_DIR="$HOME/llama.cpp/build/bin"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
PORT="${LLAMA_PORT:-8001}"
LOG="$HOME/llama-server.log"

declare -A MODELS
MODELS[minimax]="$HOME/models/MiniMax-M2.1-GGUF/UD-Q2_K_XL/MiniMax-M2.1-UD-Q2_K_XL-00001-of-00002.gguf"
MODELS[gpt-oss]="$HOME/models/gpt-oss-120b-GGUF/gpt-oss-120b-Q8_0-00001-of-00002.gguf"
MODELS[glm-flash]="$HOME/models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf"

usage() {
    echo "Usage: $0 <model-name|status|stop>"
    echo "Models: minimax, gpt-oss, glm-flash"
    echo "Set LLAMA_PORT env var to change port (default: 8001)"
    exit 1
}

get_status() {
    local pid=$(pgrep -x llama-server || true)
    if [ -z "$pid" ]; then
        echo '{"status":"stopped","model":null}'
    else
        local cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        local model="unknown"
        [[ "$cmd" == *"MiniMax"* ]] && model="minimax"
        [[ "$cmd" == *"gpt-oss"* ]] && model="gpt-oss"
        [[ "$cmd" == *"GLM"* ]] && model="glm-flash"
        echo "{\"status\":\"running\",\"model\":\"$model\",\"pid\":$pid}"
    fi
}

stop_server() {
    pkill -x llama-server 2>/dev/null || true
    sleep 2
    pkill -9 -x llama-server 2>/dev/null || true
    sleep 1
}

[ $# -lt 1 ] && usage

case "$1" in
    status) get_status; exit 0 ;;
    stop) stop_server; echo '{"status":"stopped"}'; exit 0 ;;
esac

MODEL_NAME="$1"
MODEL_PATH="${MODELS[$MODEL_NAME]}"
[ -z "$MODEL_PATH" ] && { echo "Unknown model: $MODEL_NAME"; usage; }
[ ! -f "$MODEL_PATH" ] && { echo "Model not found: $MODEL_PATH"; exit 1; }

stop_server

nohup "$LLAMA_DIR/llama-server" \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -c 4096 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --jinja \
    > "$LOG" 2>&1 &

for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
        echo "{\"status\":\"ready\",\"model\":\"$MODEL_NAME\",\"port\":$PORT}"
        exit 0
    fi
    sleep 2
done
echo '{"status":"timeout"}'
exit 1
