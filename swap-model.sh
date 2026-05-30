#!/bin/bash
set -e

LLAMA_DIR="$HOME/llama.cpp/build/bin"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
PORT="${LLAMA_PORT:-8001}"
LOG="$HOME/llama-server.log"
DEEPSEEK_CONTAINER="deepseek-vllm-backend"
N_GPU_LAYERS=99
READY_ATTEMPTS=120

declare -A MODELS
MODELS[minimax]="$HOME/models/MiniMax-M2.1-GGUF/UD-Q2_K_XL/MiniMax-M2.1-UD-Q2_K_XL-00001-of-00002.gguf"
MODELS[gpt-oss]="$HOME/models/gpt-oss-120b-GGUF/gpt-oss-120b-Q8_0-00001-of-00002.gguf"
MODELS[glm-flash]="$HOME/models/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf"
MODELS[leanstral]="$HOME/models/Leanstral-2603-GGUF/mistralai_Leanstral-128x3.9B-2603-Q4_K_M.gguf"
MODELS[kat-dev]="$HOME/models/KAT-Dev-72B-Exp-GGUF/KAT-Dev-72B-Exp.i1-Q4_K_M.gguf"
MODELS[qwen35-397b]="$HOME/models/Qwen3.5-397B-A17B-GGUF/UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"

declare -A VLLM_MODELS
VLLM_MODELS[deepseek-v4-flash]="deepseek-v4-flash"

usage() {
    echo "Usage: $0 <model-name|status|stop>"
    echo "llama.cpp models: minimax, gpt-oss, glm-flash, leanstral, kat-dev, qwen35-397b"
    echo "vLLM models: deepseek-v4-flash"
    echo "Set LLAMA_PORT env var to change port (default: 8001)"
    exit 1
}

get_status() {
    local llama_pid=$(pgrep -x llama-server || true)
    local docker_running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -x "$DEEPSEEK_CONTAINER" || true)

    if [ -n "$docker_running" ]; then
        echo "{\"status\":\"running\",\"model\":\"deepseek-v4-flash\",\"backend\":\"vllm-docker\"}"
    elif [ -n "$llama_pid" ]; then
        local cmd=$(ps -p "$llama_pid" -o args= 2>/dev/null || true)
        local model="unknown"
        [[ "$cmd" == *"MiniMax"* ]] && model="minimax"
        [[ "$cmd" == *"gpt-oss"* ]] && model="gpt-oss"
        [[ "$cmd" == *"GLM"* ]] && model="glm-flash"
        [[ "$cmd" == *"Leanstral"* ]] && model="leanstral"
        [[ "$cmd" == *"KAT-Dev"* ]] && model="kat-dev"
        [[ "$cmd" == *"Qwen3.5-397B"* ]] && model="qwen35-397b"
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

stop_deepseek() {
    docker rm -f "$DEEPSEEK_CONTAINER" >/dev/null 2>&1 || true
    local watchdog_pid_file="$HOME/deepseek-watchdog.pid"
    if [ -f "$watchdog_pid_file" ]; then
        kill "$(cat "$watchdog_pid_file")" 2>/dev/null || true
        rm -f "$watchdog_pid_file"
    fi
    sleep 2
}

stop_all() {
    stop_llama
    stop_deepseek
}

start_deepseek() {
    stop_all

    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PORT="$PORT" CONTAINER_NAME="$DEEPSEEK_CONTAINER" \
        bash "$SCRIPT_DIR/launch-deepseek.sh"
}

start_llama_model() {
    local model_name="$1"
    local model_path="${MODELS[$model_name]}"

    stop_all

    # Build model-specific flags
    local EXTRA_FLAGS=()
    case "$model_name" in
        leanstral)
            EXTRA_FLAGS+=(-fa on -fit on -c 65536)
            EXTRA_FLAGS+=(--chat-template-file "$HOME/models/Leanstral-2603-GGUF/chat_template.jinja")
            ;;
        qwen35-397b)
            N_GPU_LAYERS=64
            READY_ATTEMPTS=300
            EXTRA_FLAGS+=(-fa on -c 512 -np 1 --reasoning-format deepseek --reasoning off)
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

if [ -n "${VLLM_MODELS[$MODEL_NAME]+x}" ]; then
    start_deepseek
elif [ -n "${MODELS[$MODEL_NAME]+x}" ]; then
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    [ ! -f "$MODEL_PATH" ] && { echo "{\"status\":\"error\",\"message\":\"Model file not found: $MODEL_PATH\"}"; exit 1; }
    start_llama_model "$MODEL_NAME"
else
    echo "{\"status\":\"error\",\"message\":\"Unknown model: $MODEL_NAME\"}"
    exit 1
fi
