#!/bin/bash
# Launch GLM-5.3-Flash-Uncensored-NVFP4 on SparkLab (NVMe expert streaming).
# Started/stopped by sparklab-glm53.service, which swap-model.sh controls.
set -euo pipefail
cd /tmp   # SparkLab ships a top-level platform.py; never run from its source tree
VISION=()
[ -n "${GLM53_VISION_MODEL:-}" ] && VISION=(--vision-model "$GLM53_VISION_MODEL")
exec /opt/sparklab/venv/bin/sparklab serve \
  --model "$GLM53_MODEL" \
  --served-model-name "$GLM53_SERVED_NAME" \
  --host 127.0.0.1 --port "$GLM53_PORT" \
  --moe-backend offload --moe-storage disk --moe-host-cache-gb 0 \
  --moe-cache-auto --memory-ratio "$GLM53_MEMORY_RATIO" \
  --attention-backend dsa --nvfp4-backend triton \
  --moe-prefill-sparse-max-tokens 512 --disable-moe-prefill-overlap \
  --moe-prefill-hit-d2d --moe-shared-expert-overlap \
  --cuda-graph-max-bs 0 --page-size 1 --cache-type naive \
  --max-running-requests "$GLM53_MAX_RUNNING" \
  --max-seq-len-override "$GLM53_MAX_SEQ_LEN" \
  --reasoning-parser glm --tool-call-parser glm47 \
  "${VISION[@]}"
