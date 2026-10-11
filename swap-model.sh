#!/bin/bash
set -e

set -a
source /etc/spark/inference.env
set +a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm-registry.sh
source "$SCRIPT_DIR/vllm-registry.sh"

LLAMA_DIR="${SPARK_LLAMA_BIN}"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
PORT="${LLAMA_PORT:-8001}"
LOG="${SPARK_INFERENCE_STATE}/llama-server.log"
VLLM_CONTAINER="vllm-backend"
VLLM_CURRENT_FILE="${SPARK_INFERENCE_STATE}/vllm-current"
N_GPU_LAYERS=99
READY_ATTEMPTS=120



is_vllm_model() {
    local m="$1"
    for k in $(vllm_keys); do [ "$k" = "$m" ] && return 0; done
    return 1
}

usage() {
    echo "Usage: $0 <model-name|status|stop>"
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
        # llama.cpp models were retired; report a leftover server, if any.
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
    local watchdog_pid_file="${SPARK_INFERENCE_STATE}/vllm-watchdog.pid"
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
    # Preflight (image, weights, config) before stopping the current backend,
    # so a request for a broken model does not take down the working one.
    local preflight
    if ! preflight=$(PRINT_ONLY=1 PORT="$PORT" CONTAINER_NAME="$VLLM_CONTAINER" \
            bash "$SCRIPT_DIR/launch-vllm.sh" "$key" 2>&1); then
        echo "$preflight" | grep '^{' | tail -1 | grep . || \
            echo "{\"status\":\"error\",\"message\":\"preflight failed for $key\"}"
        return 1
    fi
    stop_all
    PORT="$PORT" CONTAINER_NAME="$VLLM_CONTAINER" \
        bash "$SCRIPT_DIR/launch-vllm.sh" "$key"
}


[ $# -lt 1 ] && usage

case "$1" in
    status) get_status; exit 0 ;;
    stop) stop_all; echo '{"status":"stopped"}'; exit 0 ;;
esac

MODEL_NAME="$1"

if is_vllm_model "$MODEL_NAME"; then
    start_vllm_model "$MODEL_NAME"
else
    echo "{\"status\":\"error\",\"message\":\"Unknown model: $MODEL_NAME\"}"
    exit 1
fi
