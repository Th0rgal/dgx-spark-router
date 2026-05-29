#!/bin/bash
set -euo pipefail

PORT=${PORT:-8001}
CONTAINER_NAME=${CONTAINER_NAME:-deepseek-vllm-backend}
HOST=${HOST:-0.0.0.0}

SPARK_ROOT=${SPARK_ROOT:-$HOME/spark}
HF_HOME=${HF_HOME:-${SPARK_ROOT}/models/hf-cache}
IMAGE=${IMAGE:-vllm-node-dsv4-cutlass451:latest}
IMAGE_REF=${IMAGE_REF:-ghcr.io/0xsero/deepseek-v4-flash-spark-vllm:cutlass451-g27}
PATCHER=${PATCHER:-${SPARK_ROOT}/serve/patch_vllm_k160_native.py}

MODEL_REPO=${MODEL_REPO:-0xSero/DeepSeek-V4-Flash-180B}
MODEL_REVISION=${MODEL_REVISION:-7c360e1cd4a5168099dbc54d16d929bf6df04990}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash}

CONTEXT_LENGTH=${CONTEXT_LENGTH:-200000}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-6G}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
THINKING=${THINKING:-true}
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG:-'{"method":"deepseek_mtp","num_speculative_tokens":2}'}
WATCHDOG_MIN_AVAILABLE_KB=${WATCHDOG_MIN_AVAILABLE_KB:-6291456}

MODEL_DIR="${HF_HOME}/models--${MODEL_REPO//\//--}/snapshots/${MODEL_REVISION}"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found at: $MODEL_DIR"
    echo "Run install-deepseek.sh first to download the model and Docker image."
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Docker image $IMAGE not found."
    echo "Run install-deepseek.sh first."
    exit 1
fi

if [ ! -f "$PATCHER" ]; then
    echo "Warning: patcher not found at $PATCHER, will attempt without patching"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

WATCHDOG_LOG="$HOME/deepseek-watchdog.log"
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
echo $! > "$HOME/deepseek-watchdog.pid"

args=(
  vllm serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --block-size 256
  --max-model-len "$CONTEXT_LENGTH"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --distributed-executor-backend mp
  --tokenizer-mode deepseek_v4
  --load-format safetensors
  --disable-uvicorn-access-log
  --enable-prefix-caching
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  --default-chat-template-kwargs "{\"thinking\":${THINKING}}"
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
)

if [ -n "$SPECULATIVE_CONFIG" ]; then
  args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

ENTRYPOINT_CMD="exec $(printf '%q ' "${args[@]}")"
if [ -f "$PATCHER" ]; then
  ENTRYPOINT_CMD="python3 '$PATCHER' && $ENTRYPOINT_CMD"
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --entrypoint bash \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$SPARK_ROOT:$SPARK_ROOT" \
  -v "$HOME:$HOME" \
  -e HF_HOME="$HF_HOME" \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1 \
  -e VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP=1 \
  -e VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP=0 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e DG_JIT_USE_NVRTC=0 \
  -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_DEBUG=WARN \
  -e K160_DISABLE_CUTEDSL=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  -lc "$ENTRYPOINT_CMD"

echo "DeepSeek container '$CONTAINER_NAME' started, waiting for health check..."

for i in $(seq 1 260); do
    if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q "ok\|healthy"; then
        echo "{\"status\":\"ready\",\"model\":\"deepseek\",\"port\":$PORT}"
        exit 0
    fi
    sleep 2
done

echo '{"status":"timeout"}'
exit 1
