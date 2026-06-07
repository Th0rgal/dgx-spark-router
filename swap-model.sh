#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm-registry.sh
source "$SCRIPT_DIR/vllm-registry.sh"

LLAMA_DIR="$HOME/llama.cpp/build/bin"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
PORT="${LLAMA_PORT:-8001}"
LOG="$HOME/llama-server.log"
VLLM_CONTAINER="vllm-backend"
VLLM_CURRENT_FILE="$HOME/.vllm-current"
N_GPU_LAYERS=99
READY_ATTEMPTS=120

declare -A MODELS
MODELS[gpt-oss]="$HOME/models/gpt-oss-120b-GGUF/gpt-oss-120b-Q8_0-00001-of-00002.gguf"
MODELS[leanstral]="$HOME/models/Leanstral-2603-GGUF/mistralai_Leanstral-128x3.9B-2603-Q4_K_M.gguf"

is_vllm_model() {
    local m="$1"
    for k in $(vllm_keys); do [ "$k" = "$m" ] && return 0; done
    return 1
}

usage() {
    echo "Usage: $0 <model-name|status|stop>"
    echo "llama.cpp models: ${!MODELS[*]}"
    echo "vLLM models: $(vllm_keys)"
    echo "Set LLAMA_PORT env var to change port (default: 8001)"
    exit 1
}

get_status() {
    local llama_pid docker_running
    llama_pid=$(pgrep -x llama-server || true)
    docker_running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -x "$VLLM_CONTAINER" || true)

    if [ -n "$docker_running" ]; then
        local key="unknown"
        [ -f "$VLLM_CURRENT_FILE" ] && key=$(cat "$VLLM_CURRENT_FILE")
        echo "{\"status\":\"running\",\"model\":\"$key\",\"backend\":\"vllm-docker\"}"
    elif [ -n "$llama_pid" ]; then
        local cmd model="unknown"
        cmd=$(ps -p "$llama_pid" -o args= 2>/dev/null || true)
        [[ "$cmd" == *"gpt-oss"* ]] && model="gpt-oss"
        [[ "$cmd" == *"Leanstral"* ]] && model="leanstral"
        echo "{\"status\":\"running\",\"model\":\"$model\",\"backend\":\"llama.cpp\",\"pid\":$llama_pid}"
    else
        echo '{"status":"stopped","model":null}'
    fi
}

stop_llama() {
    pkill -x llama-server 2>/dev/null || true
    sleep 2
    pkill -9 -x llama-server 2>/dev/null || true
    sleep 1
}

stop_vllm() {
    docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
    local watchdog_pid_file="$HOME/vllm-watchdog.pid"
    if [ -f "$watchdog_pid_file" ]; then
        kill "$(cat "$watchdog_pid_file")" 2>/dev/null || true
        rm -f "$watchdog_pid_file"
    fi
    rm -f "$VLLM_CURRENT_FILE"
    sleep 2
}

stop_all() {
    stop_llama
    stop_vllm
}

start_vllm_model() {
    local key="$1"
    stop_all
    PORT="$PORT" CONTAINER_NAME="$VLLM_CONTAINER" \
        bash "$SCRIPT_DIR/launch-vllm.sh" "$key"
}

start_llama_model() {
    local model_name="$1"
    local model_path="${MODELS[$model_name]}"

    stop_all

    local EXTRA_FLAGS=()
    case "$model_name" in
        leanstral)
            EXTRA_FLAGS+=(-fa on -fit on -c 65536)
            EXTRA_FLAGS+=(--chat-template-file "$HOME/models/Leanstral-2603-GGUF/chat_template.jinja")
            ;;
        *)
            EXTRA_FLAGS+=(-c 4096)
            ;;
    esac

    nohup "$LLAMA_DIR/llama-server" \
        -m "$model_path" \
        -ngl "$N_GPU_LAYERS" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --jinja \
        "${EXTRA_FLAGS[@]}" \
        > "$LOG" 2>&1 &

    for i in $(seq 1 "$READY_ATTEMPTS"); do
        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo "{\"status\":\"ready\",\"model\":\"$model_name\",\"port\":$PORT}"
            exit 0
        fi
        sleep 2
    done
    echo '{"status":"timeout"}'
    exit 1
}

[ $# -lt 1 ] && usage

case "$1" in
    status) get_status; exit 0 ;;
    stop) stop_all; echo '{"status":"stopped"}'; exit 0 ;;
esac

MODEL_NAME="$1"

if is_vllm_model "$MODEL_NAME"; then
    start_vllm_model "$MODEL_NAME"
elif [ -n "${MODELS[$MODEL_NAME]+x}" ]; then
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    [ ! -f "$MODEL_PATH" ] && { echo "{\"status\":\"error\",\"message\":\"Model file not found: $MODEL_PATH\"}"; exit 1; }
    start_llama_model "$MODEL_NAME"
else
    echo "{\"status\":\"error\",\"message\":\"Unknown model: $MODEL_NAME\"}"
    exit 1
fi
