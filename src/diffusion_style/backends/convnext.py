"""ConvNeXt-Large backend: a modern supervised CNN with a hierarchical pyramid.

Like VGG it yields multi-scale (B,C,H,W) stage features with no diffusion
noise, but with stronger semantics. torchvision weights (permissive license).
Stage outputs at .features.{1,3,5,7} = 64/32/16/8 px for a 256px input.
"""

from __future__ import annotations

import torch
from torchvision.models import ConvNeXt_Large_Weights, convnext_large

from .base import Backend, HookManager

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ConvNeXtBackend(Backend):
    space = "pixel"

    def __init__(self, config):
        super().__init__(config)
        self.dtype = torch.float32
        self.net = convnext_large(
            weights=ConvNeXt_Large_Weights.IMAGENET1K_V1
        ).features.eval().to(self.device)
        self.net.requires_grad_(False)
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        union = (*config.content_layers, *config.style_layers)
        self.hooks = HookManager(self.net, union)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x01 = (x + 1.0) / 2.0
        x_norm = (x01 - self.mean) / self.std
        self.hooks.reset()
        self.net(x_norm.to(self.dtype))
        return self.hooks.collect()

    def close(self) -> None:
        self.hooks.remove()
