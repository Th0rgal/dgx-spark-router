#!/bin/bash
# Launch a registry-defined model in a vLLM Docker container on DGX Spark.
# Usage: launch-vllm.sh <model-key>            (key from vllm-registry.sh)
#        PRINT_ONLY=1 launch-vllm.sh <key>     (print docker cmd, don't run)
set -euo pipefail

KEY="${1:-${VLLM_MODEL:-}}"
[ -z "$KEY" ] && { echo "Usage: $0 <model-key>" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm-registry.sh
source "$SCRIPT_DIR/vllm-registry.sh"

if ! vllm_config "$KEY"; then
    echo "{\"status\":\"error\",\"message\":\"Unknown vLLM model: $KEY\"}"
    exit 1
fi

PORT=${PORT:-8001}
HOST=${HOST:-0.0.0.0}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-backend}
SPARK_ROOT=${SPARK_ROOT:-$HOME/spark}
HF_HOME=${HF_HOME:-${SPARK_ROOT}/models/hf-cache}
PARSER_FILE=${PARSER_FILE:-$HOME/super_v3_reasoning_parser.py}
WATCHDOG_MIN_AVAILABLE_KB=${WATCHDOG_MIN_AVAILABLE_KB:-2097152}
PRINT_ONLY=${PRINT_ONLY:-0}

# Resolve the locally-cached snapshot directory (revision-agnostic).
CACHE_DIR="${HF_HOME}/models--${VR_REPO//\//--}"
MODEL_DIR=$(ls -d "${CACHE_DIR}"/snapshots/*/ 2>/dev/null | head -1 || true)
if [ -z "${MODEL_DIR}" ] || [ ! -d "${MODEL_DIR}" ]; then
    echo "Model not found under: ${CACHE_DIR}/snapshots/" >&2
    echo "{\"status\":\"error\",\"message\":\"Model $KEY not downloaded. Run install-vllm.sh $KEY first.\"}"
    exit 1
fi

if ! docker image inspect "$VR_IMAGE" >/dev/null 2>&1; then
    echo "Docker image $VR_IMAGE not found." >&2
    echo "{\"status\":\"error\",\"message\":\"Image $VR_IMAGE missing. Run install-vllm.sh $KEY first.\"}"
    exit 1
fi

# vLLM serve args, assembled from the registry config.
args=(
    vllm serve "$MODEL_DIR"
    --served-model-name "$VR_SERVED"
    --host "$HOST"
    --port "$PORT"
    --max-model-len "$VR_MAXLEN"
    --gpu-memory-utilization "$VR_GPU_UTIL"
    --kv-cache-dtype "$VR_KV_DTYPE"
    --max-num-seqs "$VR_MAXSEQS"
    "${VR_ARGS[@]}"
)

if [ -n "$VR_TOOL_PARSER" ]; then
    args+=(--enable-auto-tool-choice --tool-call-parser "$VR_TOOL_PARSER")
fi

parser_mount=()
if [ "$VR_USE_SUPERV3" = "1" ]; then
    if [ -f "$PARSER_FILE" ]; then
        parser_mount=(-v "$PARSER_FILE:/app/super_v3_reasoning_parser.py:ro")
        args+=(--reasoning-parser-plugin /app/super_v3_reasoning_parser.py --reasoning-parser super_v3)
    else
        echo "Warning: super_v3 parser not found at $PARSER_FILE; reasoning parsing disabled" >&2
    fi
elif [ -n "$VR_REASONING_PARSER" ]; then
    args+=(--reasoning-parser "$VR_REASONING_PARSER")
fi

env_flags=()
for kv in "${VR_ENV[@]}"; do
    env_flags+=(-e "$kv")
done

ENTRYPOINT_CMD="exec $(printf '%q ' "${args[@]}")"

docker_cmd=(
    docker run -d
    --name "$CONTAINER_NAME"
    --entrypoint bash
    --gpus all
    --network host
    --ipc host
    --ulimit memlock=-1 --ulimit stack=67108864
    -v "$SPARK_ROOT:$SPARK_ROOT"
    -v "$HOME:$HOME"
    "${parser_mount[@]}"
    -e HF_HOME="$HF_HOME"
    "${env_flags[@]}"
    "$VR_IMAGE"
    -lc "$ENTRYPOINT_CMD"
)

if [ "$PRINT_ONLY" = "1" ]; then
    echo "# image=$VR_IMAGE served=$VR_SERVED maxlen=$VR_MAXLEN kv=$VR_KV_DTYPE maxseqs=$VR_MAXSEQS"
    printf '%q ' "${docker_cmd[@]}"; echo
    exit 0
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Memory watchdog: if MemAvailable drops below the floor, kill the container so an
# OOM can't take the whole box down.
WATCHDOG_LOG="$HOME/vllm-watchdog.log"
: > "$WATCHDOG_LOG"
(
  echo "WATCHDOG_START $(date -Is) threshold_kb=${WATCHDOG_MIN_AVAILABLE_KB} name=${CONTAINER_NAME}"
  while true; do
    if docker ps -a --format "{{.Names}}" | grep -qx "$CONTAINER_NAME"; then
      running=$(docker inspect -f "{{.State.Running}}" "$CONTAINER_NAME" 2>/dev/null || echo false)
      if [ "$running" != true ]; then
        echo "WATCHDOG_EXIT_NOT_RUNNING $(date -Is)"
        exit 0
      fi
      available_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
      if [ "$available_kb" -lt "$WATCHDOG_MIN_AVAILABLE_KB" ]; then
        echo "WATCHDOG_KILL $(date -Is) mem_available_kb=${available_kb}"
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        exit 0
      fi
    fi
    sleep 1
  done
) >> "$WATCHDOG_LOG" 2>&1 &
echo $! > "$HOME/vllm-watchdog.pid"

"${docker_cmd[@]}" >/dev/null
echo "$KEY" > "$HOME/.vllm-current"

echo "vLLM container '$CONTAINER_NAME' ($KEY) started; waiting for health (first cold start compiles CUDA graphs and can take 15-25 min)..." >&2

# vLLM cold start on GB10 is slow; allow up to 30 min (900 * 2s).
for i in $(seq 1 900); do
    if curl -so /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null | grep -q "200"; then
        echo "{\"status\":\"ready\",\"model\":\"${VR_SERVED}\",\"key\":\"${KEY}\",\"port\":$PORT}"
        exit 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "Container '$CONTAINER_NAME' exited during startup. Last logs:" >&2
        docker logs --tail 40 "$CONTAINER_NAME" 2>&1 >&2 || true
        echo '{"status":"error","message":"container exited during startup; check docker logs"}'
        exit 1
    fi
    sleep 2
done

echo '{"status":"timeout"}'
exit 1
