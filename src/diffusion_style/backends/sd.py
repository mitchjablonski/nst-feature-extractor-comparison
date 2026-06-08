"""SD1.5 latent backend: frozen UNet features read at a fixed noised timestep,
optimization happening in VAE latent space."""

from __future__ import annotations

import torch

from .. import images
from ..model import load_models
from .base import Backend, HookManager


class SDBackend(Backend):
    space = "latent"

    def __init__(self, config):
        super().__init__(config)
        self.bundle = load_models(config)
        self.device = self.bundle.device
        self.dtype = self.bundle.dtype  # fp16 compute; the optimization latent stays fp32
        self.timestep = int(config.timestep)
        self.seed = int(config.seed)
        self.alphas_cumprod = self.bundle.scheduler.alphas_cumprod.to(self.device).float()
        union = (*config.content_layers, *config.style_layers)
        self.hooks = HookManager(self.bundle.unet, union)
        self._eps: torch.Tensor | None = None

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return images.encode_to_latent(self.bundle, image).float()

    def _fixed_noise(self, latent: torch.Tensor) -> torch.Tensor:
        if self._eps is None or self._eps.shape != latent.shape:
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
            self._eps = torch.randn(latent.shape, generator=gen, device=self.device, dtype=torch.float32)
        return self._eps

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        eps = self._fixed_noise(x)
        abar = self.alphas_cumprod[self.timestep]
        z_t = (abar.sqrt() * x + (1.0 - abar).sqrt() * eps).to(self.dtype)
        timesteps = torch.tensor([self.timestep], device=self.device)
        self.hooks.reset()
        self.bundle.unet(z_t, timesteps, encoder_hidden_states=self.bundle.text_embeddings)
        return self.hooks.collect()

    def to_pixels(self, x: torch.Tensor) -> torch.Tensor:
        return images.decode_to_tensor(self.bundle, x.to(self.dtype))

    def to_pil(self, x: torch.Tensor):
        return images.decode_latent(self.bundle, x.detach().to(self.dtype))

    def clamp_(self, x: torch.Tensor) -> None:
        pass  # latents are unbounded; don't clamp

    def close(self) -> None:
        self.hooks.remove()
