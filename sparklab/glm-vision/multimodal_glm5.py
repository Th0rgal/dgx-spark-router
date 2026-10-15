"""Bounded OpenAI image parts -> GLM-5.3-Flash patches and <|image|> placeholders.

GLM-5.3 uses ordinary 1-D text positions for image tokens (no MRoPE), so the
payload carries only pixels and grids. Ported from FreeToken's glm5_next image
processor for the dgx-spark-router SparkLab deployment.
"""
import base64
from io import BytesIO
import json
import os
from pathlib import Path

import torch

MAX_IMAGES = 4


class Glm5ImageProcessor:
    def __init__(self, source, tokenizer):
        from transformers import AutoImageProcessor
        self.config = json.loads((Path(source) / 'config.json').read_text())
        self.tokenizer = tokenizer
        self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
        self.merge = self.config['vision_config']['spatial_merge_size']
        # Token budget per image (checkpoint default 16..8000). Every image token costs a
        # slow NVMe-backed prefill on this deployment, so cap it.
        self.kwargs = {'return_tensors': 'pt'}
        for key, env in (('min_image_tokens', 'GLM53_IMAGE_MIN_TOKENS'),
                         ('max_image_tokens', 'GLM53_IMAGE_MAX_TOKENS')):
            if os.environ.get(env):
                self.kwargs[key] = int(os.environ[env])

    def prepare(self, messages, tools, kwargs):
        from PIL import Image
        images, rendered = [], []
        for message in messages:
            m = dict(message)
            if isinstance(m.get('content'), list):
                parts = []
                for part in m['content']:
                    if part.get('type') == 'text':
                        parts.append({'type': 'text', 'text': part.get('text', '')})
                    elif part.get('type') == 'image_url':
                        if len(images) >= MAX_IMAGES:
                            raise ValueError(f'At most {MAX_IMAGES} images are supported per request')
                        value = part.get('image_url')
                        url = value.get('url') if isinstance(value, dict) else value
                        if not isinstance(url, str) or not url.startswith('data:image/') or ';base64,' not in url:
                            raise ValueError('Images must be base64 data:image URLs')
                        if len(url) > 16 * 1024 * 1024:
                            raise ValueError('Encoded image exceeds 16 MiB')
                        raw = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
                        with Image.open(BytesIO(raw)) as image:
                            if image.width * image.height > 16_000_000:
                                raise ValueError('Image exceeds 16 megapixels')
                            images.append(image.convert('RGB'))
                        # the chat template renders <|begin_of_image|><|image|><|end_of_image|>
                        parts.append({'type': 'image'})
                    else:
                        raise ValueError('Only text and image_url content parts are supported')
                m['content'] = parts
            rendered.append(m)
        processed = self.processor(images=images, **self.kwargs)
        grids = processed['image_grid_thw'].to(torch.int64)
        if bool((grids[:, 0] != 1).any()):
            raise ValueError('video/temporal input is not supported')
        prompt = self.tokenizer.apply_chat_template(rendered, tools=tools, tokenize=False,
                                                   add_generation_prompt=True, **kwargs)
        marker = '<|image|>'
        if prompt.count(marker) != len(images):
            raise ValueError('Image placeholders do not match uploaded images')
        chunks = prompt.split(marker)
        prompt = chunks[0]
        for grid, suffix in zip(grids, chunks[1:], strict=True):
            prompt += marker * (int(grid.prod()) // self.merge**2) + suffix
        ids = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.int32)
        expected = sum(int(g.prod()) // self.merge**2 for g in grids)
        if int((ids == self.config['image_token_id']).sum()) != expected:
            raise ValueError('<|image|> did not tokenize to one token per placeholder')
        pixels = processed['pixel_values'].float()
        payload = {'pixels': pixels.flatten(), 'pixel_shape': list(pixels.shape),
                   'grids': grids.flatten()}
        return prompt, ids, payload
