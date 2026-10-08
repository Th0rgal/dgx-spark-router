#!/bin/bash
# Start the Cohere Transcribe NVFP4 vLLM server on 127.0.0.1:8002.
sudo -n docker rm -f vllm-asr >/dev/null 2>&1
exec sudo -n docker run -d --name vllm-asr --restart unless-stopped --entrypoint bash --gpus all --network host --ipc host \
  -v /home/th0rgal/models/cohere-transcribe-nvfp4-vllm:/models/cohere:ro \
  -v /opt/spark/inference/asr/cohere_asr.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/cohere_asr.py:ro \
  -v /home/th0rgal/.cache/vllm-asr:/root/.cache/vllm \
  -e VLLM_TARGET_DEVICE=cuda vllm-skinny-tp1:audio -lc "exec vllm serve /models/cohere --served-model-name cohere-transcribe \
  --host 127.0.0.1 --port 8002 --trust-remote-code --dtype bfloat16 --max-model-len 1024 --max-num-seqs 4 \
  --gpu-memory-utilization 0.01 --kv-cache-memory 536870912 --enforce-eager --disable-uvicorn-access-log"
