#!/bin/bash
# One-time setup for registry-defined vLLM models on DGX Spark:
# pulls the model's NGC image, downloads its NVFP4 weights, and installs the
# super_v3 reasoning parser when the model needs it.
#
# Usage: install-vllm.sh [<model-key> ...]   (default: all keys in the registry)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm-registry.sh
source "$SCRIPT_DIR/vllm-registry.sh"

SPARK_ROOT=${SPARK_ROOT:-$HOME/spark}
HF_HOME=${HF_HOME:-${SPARK_ROOT}/models/hf-cache}
PARSER_DEST=${PARSER_DEST:-$HOME/super_v3_reasoning_parser.py}

mkdir -p "${SPARK_ROOT}/models" "${SPARK_ROOT}/logs" "${HF_HOME}"

download_repo_if_needed() {
    local repo="$1"
    [ -z "$repo" ] && return 0

    local snapshot_glob="${HF_HOME}/models--${repo//\//--}/snapshots"
    if [ -d "$snapshot_glob" ] && [ -n "$(ls -A "$snapshot_glob" 2>/dev/null)" ]; then
        echo "  weights already cached under $snapshot_glob"
    else
        echo "  downloading $repo ..."
        bash "$SCRIPT_DIR/hf-download.sh" "$repo"
    fi
}

KEYS=("$@")
[ ${#KEYS[@]} -eq 0 ] && read -r -a KEYS <<< "$(vllm_keys)"

for KEY in "${KEYS[@]}"; do
    if ! vllm_config "$KEY"; then
        echo "Unknown model key: $KEY (known: $(vllm_keys))" >&2
        exit 1
    fi
    echo "=== Installing $KEY ($VR_REPO) ==="

    # 1. Reasoning parser plugin (only models that use super_v3).
    if [ "$VR_USE_SUPERV3" = "1" ]; then
        if [ -f "$SCRIPT_DIR/super_v3_reasoning_parser.py" ]; then
            install -m 0644 "$SCRIPT_DIR/super_v3_reasoning_parser.py" "$PARSER_DEST"
            echo "  parser -> $PARSER_DEST"
        else
            echo "  WARNING: super_v3 parser source missing in $SCRIPT_DIR" >&2
        fi
    fi

    # 2. vLLM Docker image (NVIDIA NGC; loads NVFP4 on GB10).
    if docker image inspect "$VR_IMAGE" >/dev/null 2>&1; then
        echo "  image $VR_IMAGE already present"
    else
        echo "  pulling $VR_IMAGE ..."
        docker pull "$VR_IMAGE"
    fi

    # 3. Model weights (and optional speculative drafter) via the shared downloader.
    if [ -z "${VR_LOCAL_DIR:-}" ]; then
        download_repo_if_needed "$VR_REPO"
    fi
    download_repo_if_needed "${VR_DRAFT_REPO:-}"
done

echo ""
echo "=== Setup complete: ${KEYS[*]} ==="
echo "Use the served name (or an alias) as the router model, e.g.:"
echo "  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${VR_SERVED}\", \"messages\": [{\"role\":\"user\",\"content\":\"Hello\"}]}'"
