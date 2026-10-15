#!/usr/bin/env python3
"""Add GLM-5.3-Flash image input to an installed SparkLab 0.1.3 (idempotent).

Usage: apply-glm-vision.py <site-packages>/sparklab
Copies multimodal/glm5.py and models/glm5_next/vision.py, then edits the
tokenizer, engine, scheduler and GLM model so --vision-model works for glm5_next.
"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent
MARK = "dgx-spark-router glm vision"


def edit(rel, old, new):
    path = root / rel
    text = path.read_text()
    if new in text:
        return
    if text.count(old) != 1:
        raise SystemExit(f"{rel}: anchor not found exactly once: {old[:70]!r}")
    path.write_text(text.replace(old, new))


shutil.copy(here / "multimodal_glm5.py", root / "multimodal" / "glm5.py")
shutil.copy(here / "glm5_next_vision.py", root / "models" / "glm5_next" / "vision.py")

# tokenizer: pick the GLM processor for a glm5_next vision source
edit("tokenizer/tokenize.py",
     """            from sparklab.multimodal.qwen import QwenImageProcessor
            self._image_processor = QwenImageProcessor(self._vision_source, self.tokenizer)""",
     f"""            import json as _json, os as _os  # {MARK}
            with open(_os.path.join(self._vision_source, 'config.json')) as _f:
                _glm = _json.load(_f).get('model_type') == 'glm5_next'
            if _glm:
                from sparklab.multimodal.glm5 import Glm5ImageProcessor
                self._image_processor = Glm5ImageProcessor(self._vision_source, self.tokenizer)
            else:
                from sparklab.multimodal.qwen import QwenImageProcessor
                self._image_processor = QwenImageProcessor(self._vision_source, self.tokenizer)""")

# engine: load the GLM tower; accept glm5_next in the --vision-model gate
edit("runtime/engine/engine.py",
     """        if config.vision_model:
            from sparklab.models.qwen3_5_moe.vision import load_vision
            load_vision(self.model, config.vision_model, self.device)""",
     f"""        if config.vision_model:  # {MARK}
            if any("Glm5Next" in a for a in config.model_config.architectures):
                from sparklab.models.glm5_next.vision import load_vision
            else:
                from sparklab.models.qwen3_5_moe.vision import load_vision
            load_vision(self.model, config.vision_model, self.device)""")
edit("runtime/engine/engine.py",
     """        if not any("Qwen3_5" in a or "Qwen3_6" in a or "Qwen3_8" in a or "Qwen4Exp" in a for a in model_config.architectures):""",
     """        if not any("Qwen3_5" in a or "Qwen3_6" in a or "Qwen3_8" in a or "Qwen4Exp" in a or "Glm5Next" in a for a in model_config.architectures):""")

# scheduler: MRoPE positions are Qwen-only; GLM payloads have none
edit("runtime/scheduler/scheduler.py",
     """                    msg.mm_positions = payload['positions'].reshape(3, -1).to(self.device)
                    msg.mm_delta = payload['delta']""",
     f"""                    if 'positions' in payload:  # {MARK}
                        msg.mm_positions = payload['positions'].reshape(3, -1).to(self.device)
                        msg.mm_delta = payload['delta']""")

# GLM model: scatter image embeddings at <|image|> positions
edit("models/glm5_next/model.py",
     """        hidden = self.embed_tokens.forward(input_ids)
        streams = hidden.unsqueeze(1).expand(-1, self.hc_mult, -1)""",
     f"""        hidden = self.embed_tokens.forward(input_ids)
        embeds = getattr(get_global_ctx().batch, "mm_embeds", None)  # {MARK}
        if embeds is not None:
            mask = input_ids == self._image_token_id
            if int(mask.sum()) != embeds.shape[0]:
                raise ValueError("Image embeddings and placeholders differ")
            hidden = hidden.masked_scatter(mask.unsqueeze(-1), embeds.to(hidden.dtype))
        streams = hidden.unsqueeze(1).expand(-1, self.hc_mult, -1)""")
print("GLM vision patch applied to", root)
