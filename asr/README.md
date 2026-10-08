# Speech-to-text on the Spark: Cohere Transcribe NVFP4

- Container `vllm-asr` (image `vllm-skinny-tp1:audio`, built from ./Dockerfile =
  base image + vllm[audio] deps), port 127.0.0.1:8002, restart unless-stopped.
  Start/recreate: `sudo bash /opt/spark/inference/asr/run-asr.sh`.
- Weights: /home/th0rgal/models/cohere-transcribe-nvfp4-vllm, produced by
  `quantize_nvfp4.py` from the official BF16 weights
  (/home/th0rgal/models/cohere-transcribe-bf16, mirror evewashere/...-ungated).
  NVFP4 W4A16, block 16, no calibration; LM head / projector / convs stay bf16.
  jeffpeng3/cohere-transcribe-03-2026-NVFP4 was NOT usable: its decoder,
  attention and norm tensors do not match the official model (cos sim ~0).
- `cohere_asr.py` is vLLM's model file, patched (mounted read-only over the
  image's copy): encoder Linear layers take quant_config (NVFP4 Marlin),
  LM head forced bf16, dtype fixes for the rel-pos attention.
- Router (/opt/spark/inference/router.py) forwards /v1/audio/* to :8002,
  streaming request and SSE response, under the GPU foreground lock.
  Public: https://spark-de79.gazella-vector.ts.net/v1/audio/transcriptions
- qwen3.8-flash-next KV cache lowered 16 -> 10 GiB in vllm-registry.sh to make room.
- Benchmark: `python3 bench.py http://127.0.0.1:8000 <label>` (needs refs.json + audio/).
