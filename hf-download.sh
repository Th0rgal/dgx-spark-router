#!/bin/bash
# Download a HuggingFace model snapshot into the shared HF cache.
# Usage: hf-download.sh <repo_id> [<repo_id> ...]
set -euo pipefail

SPARK_ROOT=${SPARK_ROOT:-$HOME/spark}
HF_HOME=${HF_HOME:-${SPARK_ROOT}/models/hf-cache}
VENV=${VENV:-${SPARK_ROOT}/tools/hf-download-venv}

[ $# -lt 1 ] && { echo "Usage: $0 <repo_id> [<repo_id> ...]" >&2; exit 1; }

if [ ! -x "${VENV}/bin/python" ]; then
    python3 -m venv "$VENV"
    "${VENV}/bin/python" -m pip install -U pip >/dev/null
    "${VENV}/bin/python" -m pip install -U huggingface_hub hf_transfer hf_xet >/dev/null
fi

TOKEN=""
TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
[ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ] && TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"

mkdir -p "$HF_HOME"
for REPO in "$@"; do
    echo "=== Downloading ${REPO} -> ${HF_HOME} ==="
    HF_HOME="$HF_HOME" \
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1 \
    HF_TOKEN="${HF_TOKEN:-$TOKEN}" \
        "${VENV}/bin/python" - "$REPO" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(
    repo_id=repo,
    cache_dir=os.environ["HF_HOME"],
    token=os.environ.get("HF_TOKEN") or None,
)
print("SNAPSHOT_PATH", path)
print("DOWNLOAD_COMPLETE", repo)
PY
done
echo "ALL_DOWNLOADS_COMPLETE"
