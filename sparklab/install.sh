#!/bin/bash
# Reproduce the GLM-5.3-Flash-Uncensored-NVFP4 deployment (SparkLab engine) on the Spark.
# Idempotent; run as th0rgal (needs sudo for the systemd unit). Steps:
#  1. SparkLab 0.1.3 venv in /opt/sparklab (release wheels, no sglang-kernel)
#  2. pins: flashinfer companions 0.6.18.post1, transformers 5.16.1
#  3. dist-port patch
#  4. weights + FTW conversion (experts copied byte-for-byte, KDA kept BF16)
#  5. launch files + systemd unit (installed, not enabled: the router starts it)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REL=https://github.com/sixteen-miles-labs/sparklab/releases/download/v0.1.3
VENV=/opt/sparklab/venv
MODELS=/home/th0rgal/models
SRC=$MODELS/GLM-5.3-Flash-Uncensored-NVFP4
FTW=$MODELS/GLM-5.3-Flash-Uncensored-NVFP4-FTW
UV="${UV:-$HOME/.local/bin/uv}"

if [ ! -x "$VENV/bin/sparklab" ]; then
  sudo install -d -o "$USER" /opt/sparklab
  "$UV" venv --python 3.12 "$VENV"
  # sglang-kernel is omitted: its cu130 aarch64 asset was republished and no longer
  # matches the index hash; the GLM-5.3 DSA/Triton path does not use it.
  "$UV" pip install --python "$VENV" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://flashinfer.ai/whl --extra-index-url https://flashinfer.ai/whl/cu130 \
    "$REL/sparklab-0.1.3-cp312-cp312-manylinux_2_27_aarch64.whl[fi]" \
    "$REL/sparklab_kernel_cache-0.1.3+cu130-py3-none-linux_aarch64.whl" \
    "flashinfer-cubin==0.6.18.post1" "flashinfer-jit-cache==0.6.18.post1"
fi
# transformers >= 5.19 renames GLM-5.3 layer types and breaks SparkLab 0.1.3's parser.
"$UV" pip install --python "$VENV" "transformers==5.16.1"
ARGS="$VENV/lib/python3.12/site-packages/sparklab/serving/args.py"
grep -q SPARKLAB_DIST_PORT "$ARGS" || patch "$ARGS" "$HERE/sparklab-dist-port.patch"

[ -f "$SRC/model.safetensors.index.json" ] || \
  hf download orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 --local-dir "$SRC"
[ -f "$FTW/freetoken_weight.json" ] || (cd /tmp && "$VENV/bin/sparklab" checkpoint \
  --model "$SRC" --out "$FTW" --moe-backend offload --nvfp4-backend triton --shard-gib 8)

sudo install -d -o th0rgal /opt/spark/inference/sparklab
sudo install -m 644 -o th0rgal "$HERE/glm53.env" /opt/spark/inference/sparklab/glm53.env
sudo install -m 755 -o th0rgal "$HERE/run-glm53.sh" /opt/spark/inference/sparklab/run-glm53.sh
sudo install -m 644 "$HERE/sparklab-glm53.service" /etc/systemd/system/sparklab-glm53.service
sudo systemctl daemon-reload
echo "done: request model glm-5.3-flash through the router (port 8000)"
