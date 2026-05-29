# DGX Spark Multi-Model Router

An OpenAI-compatible API router that automatically switches between multiple models on NVIDIA DGX Spark. Supports both llama.cpp (GGUF) and vLLM (Docker) backends.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `/v1/chat/completions`
- **Automatic model switching** — Request any model by name, router handles the swap
- **Dual backend** — llama.cpp for GGUF models, vLLM Docker for DeepSeek V4
- **Model aliases** — Use semantic names like `coding`, `scientific`, `fast`, `reasoning`
- **Zero dependencies** — Pure Python stdlib, no pip packages required

## Supported Models

| Model | Aliases | Backend | Best For |
|-------|---------|---------|----------|
| MiniMax M2.1 | `minimax`, `minimax-m2.1`, `coding`, `tool-calling` | llama.cpp | Tool calling, code generation |
| GPT-OSS 120B | `gpt-oss`, `gpt-oss-120b`, `scientific`, `writing` | llama.cpp | Scientific reasoning, reports |
| GLM-4.7 Flash | `glm-flash`, `glm-4.7-flash`, `fast` | llama.cpp | Quick queries, low memory |
| DeepSeek V4 Flash | `deepseek`, `deepseek-v4-flash`, `reasoning`, `thinking` | vLLM Docker | Deep reasoning, 200K context, tool use |

## Prerequisites

- DGX Spark with 128GB unified memory
- llama.cpp built with CUDA support (`~/llama.cpp/build/bin/llama-server`)
- GGUF model files in `~/models/`
- Docker installed (for DeepSeek V4)
- Tailscale Funnel configured (optional, for public access)

## Installation

```bash
# Clone to DGX Spark
git clone <repo> ~/dgx-spark-router
cd ~/dgx-spark-router

# Copy files to home directory
cp swap-model.sh launch-deepseek.sh ~/
chmod +x ~/swap-model.sh ~/launch-deepseek.sh
cp router.py ~/
```

### DeepSeek V4 Setup (one-time)

```bash
# Download Docker image and model weights (~180B params, MXFP4)
HF_TOKEN=... bash install-deepseek.sh
```

This will:
1. Install the vLLM REAP patcher for GB10
2. Pull the vLLM Docker image (`ghcr.io/0xsero/deepseek-v4-flash-spark-vllm:cutlass451-g27`)
3. Download the model weights from HuggingFace (`0xSero/DeepSeek-V4-Flash-180B`)

### Start the Router

```bash
# Start the backend (auto-selects model on first request)
LLAMA_PORT=8001 ~/swap-model.sh minimax

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
  -d '{"model": "coding", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Model Selection

```bash
# Use MiniMax for code
curl ... -d '{"model": "coding", ...}'

# Use GPT-OSS for writing
curl ... -d '{"model": "scientific", ...}'

# Use GLM-Flash for speed
curl ... -d '{"model": "fast", ...}'

# Use DeepSeek V4 for reasoning (swaps backend to vLLM Docker)
curl ... -d '{"model": "deepseek", ...}'
```

The router automatically:
1. Detects if a different model is needed
2. Stops the current backend (llama-server or Docker container)
3. Starts the requested model with the appropriate backend
4. Forwards your request once ready

### DeepSeek V4 Flash Details

DeepSeek V4 runs via vLLM in Docker (not llama.cpp) with:
- K160 profile: 180B parameter REAP-pruned MoE model
- FP8 MLA KV cache, 200K context window
- DeepSeek MTP speculative decoding (2 tokens)
- CUDA graph compilation for performance
- Thinking/reasoning mode with `<think>` tags
- Tool calling support
- Memory watchdog to prevent OOM

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│   Your App      │────▶│    router.py    │────▶│  llama-server (GGUF) │
│                 │     │   (port 8000)   │     │  or                  │
└─────────────────┘     └────────┬────────┘     │  vLLM Docker (DSv4)  │
                                 │              │  (port 8001)         │
                                 ▼              └──────────────────────┘
                        ┌─────────────────┐
                        │ swap-model.sh   │
                        │ (model control) │
                        └─────────────────┘
```

## Configuration

### Adding GGUF Models (llama.cpp)

Edit `swap-model.sh` to add new models:

```bash
declare -A MODELS
MODELS[minimax]="$HOME/models/MiniMax-M2.1-GGUF/..."
MODELS[gpt-oss]="$HOME/models/gpt-oss-120b-GGUF/..."
MODELS[my-new-model]="$HOME/models/my-model.gguf"
```

Then add aliases in `router.py`:

```python
MODELS = {
    # ... existing ...
    "my-new-model": "my-new-model",
    "my-alias": "my-new-model",
}
```

### DeepSeek Configuration

Edit `launch-deepseek.sh` to tune vLLM parameters:

```bash
CONTEXT_LENGTH=200000      # Max context window
KV_CACHE_MEMORY_BYTES=6G   # KV cache size
THINKING=true              # Enable reasoning mode
```

### Systemd Service

For production deployment, see `router.service`:

```bash
sudo cp router.service /etc/systemd/system/
sudo systemctl enable --now router
```

## Files

- `router.py` — OpenAI-compatible HTTP server with auto-switching
- `swap-model.sh` — Bash script to start/stop/swap models (both backends)
- `launch-deepseek.sh` — vLLM Docker launcher for DeepSeek V4 Flash
- `install-deepseek.sh` — One-time setup: pulls Docker image + downloads model
- `patch_vllm_reap_gb10.py` — vLLM patches for REAP models on GB10/SM121
- `router.service` — Systemd unit file for production deployment

## License

MIT
