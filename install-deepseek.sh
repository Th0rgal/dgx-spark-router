#!/bin/bash
set -euo pipefail

SPARK_ROOT=${SPARK_ROOT:-$HOME/spark}
HF_HOME=${HF_HOME:-${SPARK_ROOT}/models/hf-cache}
IMAGE_REF=${IMAGE_REF:-ghcr.io/0xsero/deepseek-v4-flash-spark-vllm:cutlass451-g27}
LOCAL_IMAGE=${LOCAL_IMAGE:-vllm-node-dsv4-cutlass451:latest}

MODEL_REPO=${MODEL_REPO:-0xSero/DeepSeek-V4-Flash-180B}
MODEL_REVISION=${MODEL_REVISION:-7c360e1cd4a5168099dbc54d16d929bf6df04990}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== DeepSeek V4 Flash Setup for DGX Spark ==="

mkdir -p "${SPARK_ROOT}/serve" "${SPARK_ROOT}/models" "${SPARK_ROOT}/logs"

# 1. Install the vLLM patcher
echo "[1/3] Installing vLLM patcher..."
if [ -f "$SCRIPT_DIR/patch_vllm_reap_gb10.py" ]; then
    install -m 0755 "$SCRIPT_DIR/patch_vllm_reap_gb10.py" "${SPARK_ROOT}/serve/patch_vllm_k160_native.py"
    echo "  Patcher installed to ${SPARK_ROOT}/serve/patch_vllm_k160_native.py"
else
    echo "  Warning: patch_vllm_reap_gb10.py not found in $SCRIPT_DIR"
    echo "  DeepSeek may not work correctly without the REAP GB10 patcher."
    echo "  Copy it from https://github.com/0xSero/deepseek-spark/tree/main/runtime/scripts/patch_vllm_reap_gb10.py"
fi

# 2. Pull Docker image
echo "[2/3] Setting up Docker image..."
if docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
    echo "  Image $LOCAL_IMAGE already exists"
else
    echo "  Pulling $IMAGE_REF ..."
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        echo "$GITHUB_TOKEN" | docker login ghcr.io -u "${GITHUB_USER:-0xSero}" --password-stdin >/dev/null
    fi
    docker pull "$IMAGE_REF"
    docker tag "$IMAGE_REF" "$LOCAL_IMAGE"
    echo "  Tagged as $LOCAL_IMAGE"
fi

# 3. Download model weights
echo "[3/3] Downloading model weights..."
SNAPSHOT_DIR="${HF_HOME}/models--${MODEL_REPO//\//--}/snapshots/${MODEL_REVISION}"
if [ -d "$SNAPSHOT_DIR" ]; then
    echo "  Model already cached at $SNAPSHOT_DIR"
else
    echo "  Downloading ${MODEL_REPO} (revision ${MODEL_REVISION})..."
    echo "  This may take a while for the first download."
    VENV="${SPARK_ROOT}/tools/hf-download-venv"
    if [ ! -x "${VENV}/bin/python" ]; then
        python3 -m venv "$VENV"
    fi
    "${VENV}/bin/python" -m pip install -U pip >/dev/null
    "${VENV}/bin/python" -m pip install -U huggingface_hub hf_transfer >/dev/null
    HF_HOME="$HF_HOME" HF_HUB_ENABLE_HF_TRANSFER=1 \
        "${VENV}/bin/python" -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_REPO}',
    revision='${MODEL_REVISION}',
    cache_dir=os.environ['HF_HOME'],
    token=os.environ.get('HF_TOKEN') or None,
    resume_download=True,
)
"
    echo "  Model downloaded to $SNAPSHOT_DIR"
fi

echo ""
echo "=== Setup Complete ==="
echo "Model: $MODEL_REPO"
echo "Image: $LOCAL_IMAGE"
echo "Snapshot: $SNAPSHOT_DIR"
echo ""
echo "You can now use 'deepseek' as a model name in the router:"
echo "  curl https://spark-de79.gazella-vector.ts.net/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"deepseek\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
