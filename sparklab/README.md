# GLM-5.3-Flash-Uncensored-NVFP4 via SparkLab (NVMe expert streaming)

**Inference engine: SparkLab 0.1.3 native engine (FreeToken-derived), not vLLM.**
The router exposes it as `glm-5.3-flash` (aliases `glm`, `glm-5.3`, `glm5`) on the
same OpenAI-compatible endpoint; swapping works like the other chat models.

## Why not vLLM
The Spark's vLLM image has no `glm5_next` architecture, and vLLM has no SSD
expert streaming (its CPU offload lives in the same unified memory on GB10).
The checkpoint is 192 GiB against 119 GiB of memory. SparkLab's own vLLM backend
only serves fully resident models. Native vLLM would mean porting glm5_next
plus an NVMe expert cache into vLLM's MoE layers.

## Layout
- Source: `~/models/GLM-5.3-Flash-Uncensored-NVFP4` (orcarouter, gated, 192 GiB)
- FTW: `~/models/GLM-5.3-Flash-Uncensored-NVFP4-FTW` (176 GiB, 24 shards).
  Routed experts byte-identical to the source NVFP4 (packed E2M1 + FP8 block
  scales; per-expert global stored FP16, rel. err <= 2.4e-4). KDA projections
  kept BF16 (no `--kda-quantization`); FP32 aux (A_log, dt_bias, router bias,
  mHC base/scale) kept FP32. MTP layer and vision tower not used.
- `glm53.env`, `run-glm53.sh` -> `/opt/spark/inference/sparklab/`
- `sparklab-glm53.service` -> `/etc/systemd/system/` (not enabled; the router's
  `swap-model.sh` starts/stops it, `MemorySwapMax=0`)
- Engine venv `/opt/sparklab/venv`, with `sparklab-dist-port.patch` (its torch
  rendezvous used API port+1 = 8002, which belongs to vllm-asr; now 8011).
- `install.sh` reproduces all of the above. `glmbench.py` is the probe below.

## Memory policy
`--memory-ratio 0.88` of free memory at start (recipe: 0.96): 16.6 GiB resident
weights, 3,321 expert-cache slots, 9,002 KV tokens, 9.2 GiB left free on the
GPU side, `--max-running-requests 1`, context capped at 8,192 tokens,
`--moe-host-cache-gb 0` (no host cache). No swap growth observed.

## Measured on the Spark (2026-10-15, one request, temperature 0)
| | value |
|---|---|
| Swap-in through router (qwen unloaded -> GLM ready) | 33-39 s |
| First request after load (TTFT) | 10 s |
| TTFT, new prompt (38 tok) | 8.2 s |
| TTFT, repeated prompt | 6.9 s |
| TTFT, tool prompt (167 tok) | 67 s |
| Decode | 3.5-4.9 tok/s |
| NVMe read per request (~75 output tokens) | 105-145 GiB |
| Min MemAvailable while serving | 41 GiB (page cache included) |
| Engine host RSS peak / GPU allocation | 8.5 GiB / 65 GiB |
| Swap growth | none |

Decode is I/O bound by expert-cache misses (~2 GiB NVMe per token). Raising
`GLM53_MEMORY_RATIO` toward 0.96 roughly doubles the cache (published FTW recipe:
~5-6 tok/s) at the cost of the OS/ASR reserve.

## Not supported
Vision/video input, MTP speculative decoding, >1 concurrent request, context
> ~9k tokens. Reasoning is returned in `reasoning_content`. The budget is set
with `reasoning_effort` (router maps it into `chat_template_kwargs`; GLM levels
low/high/max, default max when omitted).

## Rollback
```bash
sudo systemctl stop sparklab-glm53 && sudo rm /etc/systemd/system/sparklab-glm53.service
sudo systemctl daemon-reload
cd /opt/spark/inference && for f in router.py swap-model.sh; do sudo cp $f.bak-glm53-1015 $f; done
sudo systemctl restart dgx-router
# optional: sudo rm -rf /opt/sparklab ~/models/GLM-5.3-Flash-Uncensored-NVFP4{,-FTW}
```
Or `git revert` the commit that added this directory and redeploy the scripts.
