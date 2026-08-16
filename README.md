# DGX Spark Multi-Model Router

An OpenAI-compatible API router that automatically switches between multiple models on NVIDIA DGX Spark. Supports both llama.cpp (GGUF) and vLLM (Docker, NVFP4) backends.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `/v1/chat/completions`
- **Automatic model switching** — Request any model by name, router handles the swap
- **Dual backend** — llama.cpp for GGUF models, vLLM Docker for NVFP4 models
- **Model registry** — vLLM models are defined declaratively in `vllm-registry.sh`
- **Model aliases** — Use semantic names like `scientific`, `proving`, `reasoning`
- **Zero dependencies** — Router is pure Python stdlib, no pip packages required

## Supported Models

In the `"model"` field of a request, use the **primary name** (what `/v1/models` returns) or any of the accepted aliases. Names are case-insensitive; an unknown name is rejected with `Unknown model: <name>`.

| Display name | Primary `model` name | Other accepted aliases | Backend | Best for |
|---|---|---|---|---|
| GPT-OSS 120B | **`gpt-oss-120b`** | `gpt-oss`, `scientific`, `writing` | llama.cpp | Scientific reasoning, reports |
| Leanstral 1.5 119B A6B | **`leanstral-1.5-119b-a6b`** | `leanstral`, `leanstral-1.5`, `lean4`, `proving` | llama.cpp (GGUF NVFP4) | Lean 4 theorem proving |
| Leanstral 2603 | **`leanstral-2603`** | legacy: `leanstral-2603` | llama.cpp | Lean 4 theorem proving |
| Nemotron-3 Super 120B-A12B | **`nemotron-3-super`** | `nemotron`, `nemotron-3`, `nemotron-super`, `nemotron-3-super-120b`, `reasoning`, `thinking` | vLLM (NVFP4) | Deep reasoning, tool use (~32K context) |

| Qwen3.8 27B AEON Ultimate | **`qwen3.8-aeon-bf16`** | `qwen3.8-aeon`, `qwen3.8-27b-aeon`, `aeon-ultimate`, full HF repo ID | vLLM (BF16) | Uncensored multimodal reasoning and tool use (~64K context) |
| Gemma-4 26B-A4B | **`gemma-4`** | `gemma4`, `gemma`, `gemma-4-26b`, `gemma-4-26b-a4b` | vLLM (NVFP4) | Multimodal chat, fast, long context (~64K) |

Only one heavy backend runs at a time — the models are too large to co-reside in 128GB unified memory, so the router tears down the current backend before starting the next.

## Prerequisites

- DGX Spark (GB10 / Blackwell SM12.1) with 128GB unified memory
- llama.cpp built with CUDA support (`~/llama.cpp/build/bin/llama-server`)
- GGUF model files in `~/models/`
- Docker installed (for the vLLM / NVFP4 models)
- Tailscale Funnel configured (optional, for public access)

## Installation

```bash
# Clone to DGX Spark
git clone <repo> ~/dgx-spark-router
cd ~/dgx-spark-router

# Copy runtime files to home directory
cp router.py swap-model.sh vllm-registry.sh launch-vllm.sh ~/
chmod +x ~/swap-model.sh ~/launch-vllm.sh
```

`swap-model.sh` sources `vllm-registry.sh` and invokes `launch-vllm.sh` from its own directory, so keep the three together.


### Leanstral 1.5 model setup

Leanstral 1.5 is served through the router as `leanstral-1.5-119b-a6b`
(canonical key `leanstral-1.5`; aliases `leanstral`, `lean4`, `proving`).
The runtime file is the GB10-tested GGUF NVFP4 quantization of
[`mistralai/Leanstral-1.5-119B-A6B`](https://huggingface.co/mistralai/Leanstral-1.5-119B-A6B):

```bash
hf download Frosty40/Leanstral-1.5-119B-A6B-GGUF-NVFP4 \
  Leanstral-1.5-119B-A6B-NVFP4.gguf \
  --local-dir ~/models/Leanstral-1.5-119B-A6B-GGUF-NVFP4
```

The quantized GGUF does not embed `tokenizer.chat_template`. The router therefore ships and
loads Mistral's official `chat_template.jinja`; without it, llama.cpp falls back to an incompatible
ChatML-style prompt and returns textual pseudo-calls instead of structured `message.tool_calls`.
The vendored template includes one llama.cpp/Jinja compatibility correction: the official tool
result branch omits the required `support_thinking` macro argument. Deploy the `chat-templates/`
directory alongside `swap-model.sh`.

Then request it through the router:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"leanstral","messages":[{"role":"user","content":"Say READY in Lean style."}],"max_tokens":32}'
```

### vLLM Model Setup (one-time)

```bash
# Install all registry models (pulls NGC images, downloads NVFP4 weights,
# installs the super_v3 parser where needed)
bash install-vllm.sh

# ...or just one model key
bash install-vllm.sh nemotron-3-super
```

Registry keys include `nemotron-3-super`, `qwen3.6`, `qwen3.8-aeon-bf16`, and `gemma-4`. Each entry in `vllm-registry.sh` defines its image, exact Hugging Face repo, context length, and reasoning/tool parsers. Weights download into the shared HF cache under `~/spark/models/hf-cache` via `hf-download.sh`.

Install and request the AEON BF16 model with:

```bash
bash install-vllm.sh qwen3.8-aeon-bf16
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-aeon-bf16","messages":[{"role":"user","content":"Reply with exactly READY."}],"max_tokens":16}'
```

Base/runtime weights: [`AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16`](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16), BF16, approximately 55.6 GB. The route pins the official multi-arch vLLM image `v0.26.0-ubuntu2404`, which includes the Qwen3.5-family architecture used by this checkpoint.

### Start the Router

```bash
# Optionally pre-warm a backend (the router also does this on demand)
LLAMA_PORT=8001 ~/swap-model.sh gpt-oss

# Start the router (port 8000)
python3 ~/router.py
```

## Usage

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion (auto-selects/swaps model)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Model Selection

```bash
# GPT-OSS for scientific writing
curl ... -d '{"model": "scientific", ...}'

# Leanstral for Lean 4 proving
curl ... -d '{"model": "proving", ...}'

# Nemotron-3 Super for reasoning (swaps backend to vLLM Docker)
curl ... -d '{"model": "reasoning", ...}'

# Qwen3.8 AEON / Gemma-4 (vLLM Docker)
curl ... -d '{"model": "qwen3.8-aeon-bf16", ...}'
curl ... -d '{"model": "gemma", ...}'
```

The router automatically:
1. Detects if a different model is needed
2. Stops the current backend (llama-server or Docker container)
3. Starts the requested model with the appropriate backend
4. Forwards your request once ready

### vLLM / NVFP4 Details

The NVFP4 models run via vLLM in Docker (not llama.cpp). GB10/Blackwell-specific env defaults are baked into `vllm-registry.sh`:

- `VLLM_NVFP4_GEMM_BACKEND=marlin` — stable NVFP4 GEMM path on GB10
- `VLLM_FLASHINFER_MOE_BACKEND=latency` — throughput MoE kernels are SM120-only
- `VLLM_USE_FLASHINFER_MOE_FP4=0` — avoid the unstable FP4 MoE fastpath

`launch-vllm.sh` starts the container, writes the active key to `~/.vllm-current`, and runs a memory watchdog to prevent OOM. Nemotron-3 Super additionally loads the `super_v3` reasoning parser plus the `qwen3_coder` tool-call parser.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│   Your App      │────▶│    router.py    │────▶│  llama-server (GGUF) │
│                 │     │   (port 8000)   │     │  or                  │
└─────────────────┘     └────────┬────────┘     │  vLLM Docker (NVFP4) │
                                 │              │  (port 8001)         │
                                 ▼              └──────────────────────┘
                        ┌─────────────────┐
                        │ swap-model.sh   │
                        │ (model control) │──── sources vllm-registry.sh
                        └─────────────────┘──── invokes launch-vllm.sh
```

## Configuration

### Adding GGUF Models (llama.cpp)

Edit `swap-model.sh` to add a model path:

```bash
declare -A MODELS
MODELS[gpt-oss]="$HOME/models/gpt-oss-120b-GGUF/..."
MODELS[leanstral-1.5]="$HOME/models/Leanstral-1.5-119B-A6B-GGUF-NVFP4/Leanstral-1.5-119B-A6B-NVFP4.gguf"
MODELS[my-new-model]="$HOME/models/my-model.gguf"
```

### Adding vLLM Models (NVFP4)

Add a key to `vllm_keys()` and a case arm in `vllm_config()` inside `vllm-registry.sh`:

```bash
vllm_keys() { echo "nemotron-3-super qwen3.6 gemma-4 my-model"; }

# in vllm_config():
my-model)
    VR_REPO="org/My-Model-NVFP4"
    VR_SERVED="my-model"
    VR_IMAGE="nvcr.io/nvidia/vllm:26.05.post1-py3"
    ;;
```

Then run `bash install-vllm.sh my-model` to fetch the image and weights.

### Adding Aliases

For either backend, add aliases in `router.py`:

```python
MODELS = {
    # ... existing ...
    "my-model": "my-model",
    "my-alias": "my-model",
}
VALID_MODELS = {..., "my-model"}
MODEL_INFO = [..., {"id": "my-model", "object": "model", "canonical": "my-model"}]
```

### Systemd Service

For production deployment, see `router.service`:

```bash
sudo cp router.service /etc/systemd/system/
sudo systemctl enable --now router
```

## Files

- `router.py` — OpenAI-compatible HTTP server with auto-switching
- `swap-model.sh` — Start/stop/swap models across both backends
- `vllm-registry.sh` — Declarative registry of vLLM/NVFP4 model configs
- `launch-vllm.sh` — vLLM Docker launcher (image, parsers, memory watchdog)
- `install-vllm.sh` — One-time setup: pulls images, downloads weights, installs parser
- `hf-download.sh` — Shared HuggingFace snapshot downloader
- `super_v3_reasoning_parser.py` — vLLM reasoning parser plugin for Nemotron thinking modes
- `benchmark.py` — Throughput/latency benchmark helper
- `router.service` — Systemd unit file for production deployment

## License

MIT
