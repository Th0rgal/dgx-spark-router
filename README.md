# DGX Spark Router

OpenAI-compatible API router for the NVIDIA DGX Spark (GB10). It serves one
chat model through vLLM and speech-to-text through a second, always-on vLLM
instance.

## Models

| Primary `model` name | Aliases | Backend | Notes |
|---|---|---|---|
| **`qwen3.8-flash-next`** | `qwen`, `qwen3.8`, `qwen3.8-flash`, `flash-next`, full HF repo ID | vLLM (NVFP4, container `vllm-backend`, port 8001) | 125B/6B-active MoE, 256K context, MTP speculative decoding; cold start ~15 min |
| **`leanstral-1.5-119b-a6b`** | `leanstral`, `leanstral-1.5`, `lean4`, `proving` | llama.cpp (GGUF NVFP4, port 8001) | Lean 4 proving, tool calls via `chat-templates/leanstral-1.5.jinja`; 128K context; load ~3.6 min. Weights: `Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4` in `~/models` |
| **`cohere-transcribe`** | | vLLM (NVFP4 W4A16, container `vllm-asr`, port 8002) | `/v1/audio/transcriptions` only; see [`asr/README.md`](asr/README.md) |

An unknown `model` is rejected with `Unknown model: <name>`. A request without
`model` uses `qwen3.8-flash-next`. Audio requests never swap the chat model.

Only one chat model is loaded at a time; switching between qwen3.8-flash-next
and Leanstral 1.5 takes the load time of the target model.

GPT-OSS 120B, Leanstral 2603, Nemotron-3 Super, Gemma-4 and Qwen3.8 27B
OrcaRouter were retired in October 2026 (see git history to
restore one).

## Layout on the Spark

- `/opt/spark/inference`: `router.py`, `swap-model.sh`, `vllm-registry.sh`,
  `launch-vllm.sh`, `asr/` (installed copies of this repo).
- `/etc/spark/inference.env`: model and state paths.
- `/var/lib/spark/inference`: current model, watchdog log.
- `dgx-router.service` runs `router.py` on port 8000; public via Tailscale
  Funnel at `https://spark-de79.gazella-vector.ts.net`.

## Usage

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Hello"}]}'

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@clip.wav -F model=cohere-transcribe -F language=fr   # -F stream=true for SSE
```

Chat and audio requests take a shared GPU foreground lock
(`arbiter/gpu_idle.py` in dgx-spark-arbiter) so background GPU work yields.

## How it works

- `router.py`: stdlib HTTP server. Resolves aliases, asks `swap-model.sh` to
  (re)start the chat backend if needed, streams chat SSE, proxies `/v1/audio/*`
  to port 8002.
- `swap-model.sh`: start/stop/status of the chat backend. A model is
  preflighted (`launch-vllm.sh` with `PRINT_ONLY=1`) before the running one is
  stopped, so a broken entry cannot take down a working model.
- `vllm-registry.sh`: declarative vLLM model entries (image, weights, flags).
- `launch-vllm.sh`: `docker run` for a registry entry, plus a MemAvailable
  watchdog that removes the container below 2 GiB to avoid a host OOM.
- `install-vllm.sh`, `hf-download.sh`: fetch images and weights for an entry.

### Adding a vLLM model

Add its key to `vllm_keys()` and a case arm in `vllm_config()` in
`vllm-registry.sh`, run `bash install-vllm.sh <key>`, then add the key and
aliases to `MODELS`, `VALID_MODELS` and `MODEL_INFO` in `router.py`.

## Files

- `router.py`, `swap-model.sh`, `vllm-registry.sh`, `launch-vllm.sh`
- `install-vllm.sh`, `hf-download.sh`
- `super_v3_reasoning_parser.py`: vLLM reasoning-parser plugin, loaded only for
  registry entries that set `VR_REASONING_PARSER=super_v3`
- `chat-templates/leanstral-1.5.jinja`: Mistral's official template (the GGUF has none)
- `asr/`: Cohere Transcribe NVFP4 service
- `benchmark.py`, `router.service`

## License

MIT
