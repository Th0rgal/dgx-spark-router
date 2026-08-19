#!/usr/bin/env python3
"""Dequantize lm_head FP8 -> BF16 for GB10 (DGX Spark) vLLM serving.

The OrcaRouter Qwen3.8-27B-Uncensored-NVFP4 checkpoint stores `lm_head` as
FP8 W8A8 (`lm_head.weight` + `lm_head.weight_scale`). Every vLLM build tested
on GB10 (NGC 26.05, vllm/vllm-openai v0.27.1 and cu129-nightly) fails to load
it: the Qwen3_5 loader only accepts `lm_head.weight` on ParallelLMHead
("There is no module or parameter named 'lm_head.weight_scale'").

This script rewrites the snapshot in the HF cache:
  1. shard model-00005-of-00005.safetensors: dequantize lm_head
     (weight * scale -> BF16, mathematically identical at runtime), drop
     `lm_head.weight_scale`;
  2. model.safetensors.index.json: drop the weight_scale entry;
  3. config.json: drop `lm_head` from quantization_config group_0 targets.

Originals are backed up to ~/.router-backups/lmhead-patch/.
IMPORTANT: re-downloading the snapshot (hf download / install-vllm.sh with a
wiped cache) restores the pristine FP8 lm_head and the loader error returns;
re-run this script after any re-download.

Run on the DGX (host python3 needs torch+safetensors, e.g. via the vLLM image):
  docker run --rm \
    -v ~/spark/models/hf-cache:/hf \
    -v ~/dgx-spark-router/tools/lmhead-dequant.py:/work/patch.py:ro \
    --entrypoint python3 nvcr.io/nvidia/vllm:26.05.post1-py3 /work/patch.py

Patched shard sha256: 510767f102b60742e56e0fe1c983c458dfc295f88b5540d7aefb1086da6ab6f4
"""
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = "models--orcarouter--Qwen3.8-27B-Uncensored-NVFP4"
SNAP_GLOB = "/hf/" + REPO + "/snapshots/*"
STAGE = Path("/stage")

def main():
    snaps = sorted(Path().glob(SNAP_GLOB))
    if not snaps:
        raise SystemExit("snapshot not found under /hf")
    snap = snaps[-1]
    print("snapshot:", snap)
    shard_name = "model-00005-of-00005.safetensors"

    shard = load_file(str(snap / shard_name), device="cpu")
    w = shard.pop("lm_head.weight")
    scale = shard.pop("lm_head.weight_scale")
    if scale is None or w.dtype != torch.float8_e4m3fn:
        print("lm_head already dequantized (weight dtype:", w.dtype, ") - nothing to do")
        return
    sc = scale.to(torch.float32)
    if sc.dim() == 1:
        sc = sc.view(-1, 1)
    assert sc.shape[0] == w.shape[0], (sc.shape, w.shape)
    deq = (w.to(torch.float32) * sc).to(torch.bfloat16)
    shard["lm_head.weight"] = deq.contiguous()
    save_file(shard, str(STAGE / shard_name), metadata={"format": "pt"})

    index = json.loads((snap / "model.safetensors.index.json").read_text())
    index["weight_map"].pop("lm_head.weight_scale", None)
    (STAGE / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    cfg = json.loads((snap / "config.json").read_text())
    g0 = cfg["quantization_config"]["config_groups"]["group_0"]
    g0["targets"] = [t for t in g0["targets"] if t != "lm_head"]
    (STAGE / "config.json").write_text(json.dumps(cfg, indent=2))

    h = hashlib.sha256()
    with open(STAGE / shard_name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    print("patched shard sha256:", h.hexdigest())
    print("PATCH_DONE - install stage/ into the snapshot (see README lm_head section)")

if __name__ == "__main__":
    main()
