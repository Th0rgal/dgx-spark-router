"""Quantize the official Cohere Transcribe BF16 checkpoint to ModelOpt-style
NVFP4 W4A16 (weight-only, no calibration), keeping the NeMo tensor names that
vLLM's cohere_asr loader expects.

Per Linear weight W [N, K]:
  weight_scale_2 = amax(W) / (6 * 448)                       (fp32, per tensor)
  weight_scale   = fp8_e4m3(amax(block16) / 6 / weight_scale_2)  [N, K/16]
  weight         = E2M1(W / (weight_scale * weight_scale_2)), two per byte,
                   low nibble = even column, bit 3 = sign          [N, K/2] uint8
Groups vLLM fuses into one layer (decoder self-attn q/k/v, cross-attn k/v)
share one weight_scale_2 so the fused GEMM stays exact.
Kept in bf16: LM head, encoder_decoder_proj, convolutions, embeddings, norms.
"""
import json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

QUANT = re.compile(
    r"^(encoder\.layers\.\d+\.(self_attn\.linear_(q|k|v|out|pos)|feed_forward[12]\.linear[12])"
    r"|encoder\.pre_encode\.out"
    r"|transf_decoder\._decoder\.layers\.\d+\.(first_sub_layer\.(query|key|value)_net|first_sub_layer\.out_projection"
    r"|second_sub_layer\.(query|key|value)_net|second_sub_layer\.out_projection|third_sub_layer\.dense_(in|out)))\.weight$"
)


def quantize(w, gscale):
    w = w.float()
    n, k = w.shape
    blocks = w.reshape(n, k // 16, 16)
    s = (blocks.abs().amax(-1) / 6.0 / gscale).clamp(min=2**-9, max=448.0).to(torch.float8_e4m3fn)
    scaled = blocks / (s.float().unsqueeze(-1) * gscale)
    mag = scaled.abs().clamp(max=6.0)
    idx = (mag.unsqueeze(-1) - E2M1).abs().argmin(-1)          # nearest E2M1 magnitude
    nib = (idx | ((scaled < 0) & (idx > 0)).long() << 3).to(torch.uint8).reshape(n, k)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous(), s.contiguous()


f = safe_open(os.path.join(SRC, "model.safetensors"), "pt")
keys = list(f.keys())
qkeys = [k for k in keys if QUANT.match(k)]
amax = {k: f.get_tensor(k).float().abs().max() for k in qkeys}

# shared global scale for fused groups
group_of = {}
for k in qkeys:
    m = re.match(r"(transf_decoder\._decoder\.layers\.\d+\.)(first|second)_sub_layer\.(query|key|value)_net\.weight", k)
    if m and not (m.group(2) == "second" and m.group(3) == "query"):
        group_of[k] = m.group(1) + m.group(2)
gmax = {}
for k, g in group_of.items():
    gmax[g] = max(gmax.get(g, torch.tensor(0.0)), amax[k])

out, err = {}, []
for k in keys:
    t = f.get_tensor(k)
    if k in amax:
        a = gmax[group_of[k]] if k in group_of else amax[k]
        g = (a / (6.0 * 448.0)).to(torch.float32)
        q, s = quantize(t, g)
        base = k[: -len("weight")]
        out[k], out[base + "weight_scale"], out[base + "weight_scale_2"] = q, s, g.reshape(())
        # reconstruction error check
        lo, hi = q & 0x0F, q >> 4
        nib = torch.stack((lo, hi), -1).reshape(q.shape[0], -1)
        deq = (E2M1[(nib & 7).long()] * torch.where((nib & 8) > 0, -1.0, 1.0)).reshape(q.shape[0], -1, 16)
        deq = (deq * s.float().unsqueeze(-1) * g).reshape(t.shape)
        err.append(((deq - t.float()).norm() / t.float().norm()).item())
    else:
        out[k] = t.contiguous()

os.makedirs(DST, exist_ok=True)
save_file(out, os.path.join(DST, "model.safetensors"), metadata={"format": "pt"})
for n in os.listdir(SRC):
    p = os.path.join(SRC, n)
    if n != "model.safetensors" and os.path.isfile(p):
        shutil.copy(p, DST)
json.dump({"quant_method": "modelopt_fp4",
           "quantization": {"quant_algo": "W4A16_NVFP4", "kv_cache_quant_algo": None, "group_size": 16,
                            "exclude_modules": ["proj_out", "log_softmax*"]}},
          open(os.path.join(DST, "hf_quant_config.json"), "w"), indent=2)
err.sort()
print(f"{len(qkeys)} linears quantized, {len(gmax)} fused groups; rel err median {err[len(err)//2]:.4f} max {err[-1]:.4f}")
