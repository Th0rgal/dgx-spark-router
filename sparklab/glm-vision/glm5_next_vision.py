"""GLM-5.3-Flash vision tower for SparkLab: the transformers reference encoder,
loaded from the source checkpoint's ``model.visual.*`` tensors (BF16, unchanged).
Ported for the dgx-spark-router SparkLab deployment (FreeToken serves the same
tower natively; SparkLab 0.1.3 dropped it)."""
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_vision(model, source, device):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextVisionModel, Glm5NextVisionRotaryEmbedding,
    )

    source = Path(source)
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'glm5_next':
        raise ValueError('GLM-5.3 vision needs the glm5_next source checkpoint as --vision-model')
    vc = Glm5NextVisionConfig(**config['vision_config'])
    vc._attn_implementation = 'sdpa'
    if vc.out_hidden_size != model.model.embed_tokens.weight.shape[1]:
        raise ValueError('Vision output width does not match the text checkpoint')
    with torch.device('meta'):
        tower = Glm5NextVisionModel(vc)
    expected = tower.state_dict()
    index = json.loads((source / 'model.safetensors.index.json').read_text())['weight_map']
    by_file = {}
    for raw, file in index.items():
        if raw.startswith('model.visual.'):
            by_file.setdefault(file, []).append(raw)
    weights = {}
    for file, names in sorted(by_file.items()):
        with safe_open(str(source / file), framework='pt', device='cpu') as f:
            for raw in names:
                name = raw[len('model.visual.'):]
                if name not in expected:
                    raise ValueError(f'Unexpected vision tensor: {raw}')
                tensor = f.get_tensor(raw)
                if tensor.shape != expected[name].shape:
                    raise ValueError(f'Vision tensor shape mismatch: {raw}')
                weights[name] = tensor.to(device=device, dtype=torch.bfloat16)
    missing = sorted(set(expected) - set(weights))
    if missing:
        raise ValueError(f'Missing vision tensors: {missing[:5]}')
    tower.load_state_dict(weights, strict=True, assign=True)
    tower.eval()
    # Nonpersistent rotary buffers are not checkpoint tensors.
    tower.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(vc.hidden_size // vc.num_heads // 2).to(device)
    model._vision = tower
    model.model._image_token_id = config['image_token_id']
