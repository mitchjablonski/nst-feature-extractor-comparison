"""DINOv3-ConvNeXt backend: self-supervised CNN features with strong semantics.

Uses the HF backbone interface, which returns multi-scale (B,C,H,W) feature_maps
per stage -- the same VGG-shaped interface our Gram/content code expects, with no
diffusion noise. The model is a GATED HF repo: accept the license and authenticate
(huggingface-cli login) before first use, or loading raises a 401.
"""

from __future__ import annotations

import torch

from .base import Backend

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DINOv3Backend(Backend):
    space = "pixel"

    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoBackbone

        self.dtype = torch.float32
        self.model = (
            AutoBackbone.from_pretrained(config.dinov3_model_id, out_indices=(0, 1, 2, 3))
            .eval()
            .to(self.device)
        )
        self.model.requires_grad_(False)
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x01 = (x + 1.0) / 2.0
        x_norm = ((x01 - self.mean) / self.std).to(self.dtype)
        out = self.model(x_norm)
        # feature_maps is a tuple of (B, C, H, W), one per requested stage.
        return {f"stage{i}": fmap for i, fmap in enumerate(out.feature_maps)}

    def close(self) -> None:
        pass
