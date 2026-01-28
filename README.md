# DGX Spark Multi-Model Router

An OpenAI-compatible API router that automatically switches between multiple GGUF models on NVIDIA DGX Spark.

## Features

- **OpenAI-compatible API** — Drop-in replacement for `/v1/chat/completions`
- **Automatic model switching** — Request any model by name, router handles the swap
- **Model aliases** — Use semantic names like `coding`, `scientific`, `fast`
- **Zero dependencies** — Pure Python stdlib, no pip packages required

## Supported Models

| Model | Aliases | Size | Speed | Best For |
|-------|---------|------|-------|----------|
| MiniMax M2.1 | `minimax`, `minimax-m2.1`, `coding`, `tool-calling`, `orchestrator` | 86 GB | 33 t/s | Tool calling, code generation |
| GPT-OSS 120B | `gpt-oss`, `gpt-oss-120b`, `scientific`, `writing`, `reasoning` | 63 GB | 57 t/s | Scientific reasoning, reports |
| GLM-4.7 Flash | `glm-flash`, `glm-4.7-flash`, `fast`, `quick` | 18 GB | 62 t/s | Quick queries, low memory |

## Prerequisites

- DGX Spark with 128GB unified memory
- llama.cpp built with CUDA support (`~/llama.cpp/build/bin/llama-server`)
- GGUF model files in `~/models/`
- Tailscale Funnel configured (optional, for public access)

## Installation

```bash
# Clone to DGX Spark
git clone <repo> ~/dgx-spark-router
cd ~/dgx-spark-router

# Copy files to home directory
cp swap-model.sh ~/
cp router.py ~/
chmod +x ~/swap-model.sh ~/router.py

# Start the backend (llama-server on port 8001)
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

Simply specify the model name in your request:

```bash
# Use MiniMax for code
curl ... -d '{"model": "coding", ...}'

# Use GPT-OSS for writing
curl ... -d '{"model": "scientific", ...}'

# Use GLM-Flash for speed
curl ... -d '{"model": "fast", ...}'
```

The router automatically:
1. Detects if a different model is needed
2. Stops the current llama-server
3. Starts the requested model
4. Forwards your request once ready

### Swap Times

| Transition | Time |
|------------|------|
| → GLM-Flash | ~8s |
| → GPT-OSS | ~15s |
| → MiniMax | ~17s |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your App      │────▶│    router.py    │────▶│  llama-server   │
│                 │     │   (port 8000)   │     │   (port 8001)   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ swap-model.sh   │
                        │ (model control) │
                        └─────────────────┘
```

## Configuration

### Adding Models

Edit `swap-model.sh` to add new models:

```bash
declare -A MODELS
MODELS[minimax]="$HOME/models/MiniMax-M2.1-GGUF/..."
MODELS[gpt-oss]="$HOME/models/gpt-oss-120b-GGUF/..."
MODELS[my-new-model]="$HOME/models/my-model.gguf"  # Add here
```

Then add aliases in `router.py`:

```python
MODELS = {
    # ... existing ...
    "my-new-model": "my-new-model",
    "my-alias": "my-new-model",
}
```

### Systemd Service

For production deployment, see `router.service`:

```bash
sudo cp router.service /etc/systemd/system/
sudo systemctl enable --now router
```

## Files

- `router.py` — OpenAI-compatible HTTP server with auto-switching
- `swap-model.sh` — Bash script to start/stop/swap models
- `router.service` — Systemd unit file for production deployment

## License

MIT
