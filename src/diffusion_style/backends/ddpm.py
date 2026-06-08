"""Pixel-space DDPM backend: frozen unconditional pixel UNet, features read at a
fixed noised timestep, optimization happening directly on pixels (no VAE)."""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler, UNet2DModel

from .base import Backend, HookManager


class DDPMBackend(Backend):
    space = "pixel"

    def __init__(self, config):
        super().__init__(config)
        self.dtype = getattr(torch, config.dtype)
        self.unet = UNet2DModel.from_pretrained(
            config.ddpm_model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.scheduler = DDPMScheduler.from_pretrained(config.ddpm_model_id)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device).float()
        self.timestep = int(config.timestep)
        self.seed = int(config.seed)
        union = (*config.content_layers, *config.style_layers)
        self.hooks = HookManager(self.unet, union)
        self._eps: torch.Tensor | None = None

    def _fixed_noise(self, image: torch.Tensor) -> torch.Tensor:
        if self._eps is None or self._eps.shape != image.shape:
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
            self._eps = torch.randn(image.shape, generator=gen, device=self.device, dtype=torch.float32)
        return self._eps

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        eps = self._fixed_noise(x)
        abar = self.alphas_cumprod[self.timestep]
        x_t = (abar.sqrt() * x + (1.0 - abar).sqrt() * eps).to(self.dtype)
        timesteps = torch.tensor([self.timestep], device=self.device)
        self.hooks.reset()
        self.unet(x_t, timesteps)
        return self.hooks.collect()

    def close(self) -> None:
        self.hooks.remove()
