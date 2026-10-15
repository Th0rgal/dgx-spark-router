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
READY_ATTEMPTS=300   # x2 s; a 67 GB GGUF needs ~4 min from a cold page cache

# llama.cpp (GGUF) models
declare -A MODELS
MODELS[leanstral-1.5]="${SPARK_GGUF_ROOT}/Leanstral-1.5-119B-A6B-GGUF-NVFP4/Leanstral-1.5-119B-A6B-NVFP4.gguf"

# SparkLab native engine (NVMe expert streaming); model key -> systemd unit.
declare -A SPARKLAB_UNITS
SPARKLAB_UNITS[glm-5.3-flash]="sparklab-glm53.service"
SPARKLAB_READY_ATTEMPTS=900   # x2 s



is_vllm_model() {
    local m="$1"
    for k in $(vllm_keys); do [ "$k" = "$m" ] && return 0; done
    return 1
}

usage() {
    echo "Usage: $0 <model-name|status|stop>"
    echo "llama.cpp models: ${!MODELS[*]}"
    echo "vLLM models: $(vllm_keys)"
    echo "SparkLab models: ${!SPARKLAB_UNITS[*]}"
    echo "Set LLAMA_PORT env var to change port (default: 8001)"
    exit 1
}

get_status() {
    local llama_pid docker_running
    llama_pid=$(pgrep -x llama-server || true)
    docker_running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -x "$VLLM_CONTAINER" || true)

    local key
    for key in "${!SPARKLAB_UNITS[@]}"; do
        if systemctl is-active -q "${SPARKLAB_UNITS[$key]}"; then
            echo "{\"status\":\"running\",\"model\":\"$key\",\"backend\":\"sparklab\"}"
            return
        fi
    done

    if [ -n "$docker_running" ]; then
        local key="unknown"
        [ -f "$VLLM_CURRENT_FILE" ] && key=$(cat "$VLLM_CURRENT_FILE")
        echo "{\"status\":\"running\",\"model\":\"$key\",\"backend\":\"vllm-docker\"}"
    elif [ -n "$llama_pid" ]; then
        local cmd model="unknown"
        cmd=$(ps -p "$llama_pid" -o args= 2>/dev/null || true)
        [[ "$cmd" == *"Leanstral-1.5"* ]] && model="leanstral-1.5"
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

stop_sparklab() {
    local unit
    for unit in "${SPARKLAB_UNITS[@]}"; do
        systemctl is-active -q "$unit" && sudo -n systemctl stop "$unit"
    done
    return 0
}

stop_all() {
    stop_llama
    stop_vllm
    stop_sparklab
}

start_sparklab_model() {
    local key="$1" unit="${SPARKLAB_UNITS[$1]}"
    # Preflight before stopping the current backend.
    local envfile="$SCRIPT_DIR/sparklab/glm53.env" model
    model=$(sed -n 's/^GLM53_MODEL=//p' "$envfile" 2>/dev/null)
    if [ ! -x /opt/sparklab/venv/bin/sparklab ] || [ ! -f "$model/freetoken_weight.json" ]; then
        echo "{\"status\":\"error\",\"message\":\"SparkLab or prepared FTW checkpoint missing for $key\"}"
        return 1
    fi
    stop_all
    sudo -n systemctl start "$unit"
    for i in $(seq 1 "$SPARKLAB_READY_ATTEMPTS"); do
        # The SparkLab frontend answers /health at once with status "loading";
        # only "ok" means the engine finished loading.
        if curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status": *"ok"'; then
            echo "{\"status\":\"ready\",\"model\":\"$key\",\"port\":$PORT}"
            exit 0
        fi
        systemctl is-active -q "$unit" || {
            echo "{\"status\":\"error\",\"message\":\"$unit exited, see ${SPARK_INFERENCE_STATE}/sparklab-glm53.log\"}"
            exit 1
        }
        sleep 2
    done
    sudo -n systemctl stop "$unit"
    echo '{"status":"timeout"}'
    exit 1
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


start_llama_model() {
    local model_name="$1"
    local model_path="${MODELS[$model_name]}"
    local EXTRA_FLAGS=()
    case "$model_name" in
        leanstral-1.5)
            # Leanstral 1.5 119B A6B, GGUF NVFP4 (Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4).
            # The GGUF has no tokenizer.chat_template; use Mistral's official template so
            # llama.cpp formats tools and parses [TOOL_CALLS] into message.tool_calls.
            # One slot to maximize per-request context (LEANSTRAL15_CTX / _PARALLEL override).
            [ -f "$SCRIPT_DIR/chat-templates/leanstral-1.5.jinja" ] || {
                echo '{"status":"error","message":"Leanstral 1.5 chat template is missing"}'
                return 1
            }
            EXTRA_FLAGS+=(-fa on -fit on -c "${LEANSTRAL15_CTX:-131072}" -np "${LEANSTRAL15_PARALLEL:-1}")
            EXTRA_FLAGS+=(--alias leanstral-1.5)
            EXTRA_FLAGS+=(--chat-template-file "$SCRIPT_DIR/chat-templates/leanstral-1.5.jinja")
            ;;
    esac

    stop_all

    nohup "$LLAMA_DIR/llama-server" \
        -m "$model_path" \
        -ngl "$N_GPU_LAYERS" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --jinja \
        "${EXTRA_FLAGS[@]}" \
        > "$LOG" 2>&1 &

    for i in $(seq 1 "$READY_ATTEMPTS"); do
        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok"; then
            echo "{\"status\":\"ready\",\"model\":\"$model_name\",\"port\":$PORT}"
            exit 0
        fi
        pgrep -x llama-server >/dev/null || {
            echo "{\"status\":\"error\",\"message\":\"llama-server exited, see $LOG\"}"
            exit 1
        }
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
elif [ -n "${SPARKLAB_UNITS[$MODEL_NAME]+x}" ]; then
    start_sparklab_model "$MODEL_NAME"
elif [ -n "${MODELS[$MODEL_NAME]+x}" ]; then
    [ -f "${MODELS[$MODEL_NAME]}" ] || { echo "{\"status\":\"error\",\"message\":\"Model file not found: ${MODELS[$MODEL_NAME]}\"}"; exit 1; }
    start_llama_model "$MODEL_NAME"
else
    echo "{\"status\":\"error\",\"message\":\"Unknown model: $MODEL_NAME\"}"
    exit 1
fi
