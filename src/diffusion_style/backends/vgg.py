"""VGG-19 backend: the classic Gatys feature extractor and our reference baseline.

A clean feed-forward CNN — no diffusion timestep/noise. Multi-scale conv layers
(conv1_1..conv5_1) are the proven style/content lens everything else is measured
against.
"""

from __future__ import annotations

import torch
from torchvision.models import VGG19_Weights, vgg19

from .base import Backend, HookManager

# ImageNet normalization VGG was trained with.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VGGBackend(Backend):
    space = "pixel"

    def __init__(self, config):
        super().__init__(config)
        self.dtype = torch.float32
        self.net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(self.device)
        self.net.requires_grad_(False)
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        union = (*config.content_layers, *config.style_layers)
        self.hooks = HookManager(self.net, union)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x01 = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        x_norm = (x01 - self.mean) / self.std
        self.hooks.reset()
        self.net(x_norm.to(self.dtype))
        return self.hooks.collect()

    def close(self) -> None:
        self.hooks.remove()
