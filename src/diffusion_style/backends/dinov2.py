"""DINOv2-Large (ViT) backend — Apache-2.0, ungated stand-in for DINOv3.

ViT tokens are not a native multi-scale pyramid, so we read several transformer
LAYERS (output_hidden_states) and reshape each layer's patch tokens into a
(B, C, H, W) grid for the Gram/content losses. One spatial scale per layer (the
known ViT-vs-CNN limitation); layer depth stands in for VGG's scale pyramid.
Input must be a multiple of the patch size (14); preset runs at 224.
"""

from __future__ import annotations

import torch

from .base import Backend

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DINOv2Backend(Backend):
    space = "pixel"

    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoModel

        self.dtype = torch.float32
        self.model = AutoModel.from_pretrained(config.dinov2_model_id).eval().to(self.device)
        self.model.requires_grad_(False)
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        self.patch = self.model.config.patch_size
        # tokens to skip before the patch grid: CLS (+ any register tokens).
        self.skip = 1 + getattr(self.model.config, "num_register_tokens", 0)
        self.layers = list(dict.fromkeys((*config.content_layers, *config.style_layers)))

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x01 = (x + 1.0) / 2.0
        x_norm = ((x01 - self.mean) / self.std).to(self.dtype)
        out = self.model(x_norm, output_hidden_states=True)
        hs = out.hidden_states  # tuple: [embeddings, block_1, ..., block_N]
        h = x.shape[-2] // self.patch
        w = x.shape[-1] // self.patch
        feats = {}
        for name in self.layers:
            idx = int(name[len("layer"):])  # "layer12" -> 12
            tokens = hs[idx][:, self.skip:, :]  # [B, h*w, C]
            b, _, c = tokens.shape
            feats[name] = tokens.transpose(1, 2).reshape(b, c, h, w)
        return feats

    def close(self) -> None:
        pass
