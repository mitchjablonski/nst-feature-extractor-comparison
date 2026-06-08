"""Image <-> latent conversion, isolated so SD's scaling factor lives in one place."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .model import SDBundle

# SD1.5's VAE latents are scaled by this constant before the UNet sees them.
LATENT_SCALING = 0.18215


def load_image(path: str, resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Load an image as a [1, 3, H, W] tensor in [-1, 1] (the VAE's input range)."""
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W] in [0, 1]
    t = t * 2.0 - 1.0
    return t.to(device=device, dtype=dtype)


@torch.no_grad()
def encode_to_latent(bundle: SDBundle, image: torch.Tensor) -> torch.Tensor:
    """VAE-encode to a latent. Uses the distribution mean (deterministic)."""
    posterior = bundle.vae.encode(image).latent_dist
    return posterior.mean * LATENT_SCALING


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert an image tensor in [-1, 1] ([1, 3, H, W]) to PIL.

    The pixel-space backend optimizes the image directly, so its 'decode' is
    just this clamp+to-uint8 (no VAE).
    """
    image = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def decode_to_tensor(bundle: SDBundle, latent: torch.Tensor) -> torch.Tensor:
    """Differentiable decode: latent -> image tensor in ~[0, 1], [1, 3, H, W].

    Unlike decode_latent this keeps the graph so a pixel-space loss (TV) can
    backprop through the VAE decoder to the latent. No clamp, to keep gradients
    clean at the range edges.
    """
    image = bundle.vae.decode(latent / LATENT_SCALING).sample
    return (image + 1.0) / 2.0


@torch.no_grad()
def decode_latent(bundle: SDBundle, latent: torch.Tensor) -> Image.Image:
    """Decode a latent back to a PIL image."""
    image = bundle.vae.decode(latent / LATENT_SCALING).sample
    image = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
    arr = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)
