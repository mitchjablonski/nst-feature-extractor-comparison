"""Load the frozen Stable Diffusion 1.5 components used as a feature extractor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .config import Config


@dataclass
class SDBundle:
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    scheduler: DDPMScheduler
    text_embeddings: torch.Tensor  # empty-prompt embedding, [1, 77, 768]
    device: torch.device
    dtype: torch.dtype


def load_models(config: Config) -> SDBundle:
    """Load VAE + UNet + text encoder + scheduler, frozen, in the configured dtype.

    All params are frozen (we never train the model); we only backprop to the
    latent. Gradient checkpointing on the UNet is what lets autograd through a
    full UNet forward fit in 8GB.
    """
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet", torch_dtype=dtype).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        config.model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")

    for module in (vae, unet, text_encoder):
        module.eval()
        module.requires_grad_(False)

    # Recompute activations in backward instead of storing them all -> fits 8GB.
    unet.enable_gradient_checkpointing()

    # Unconditional (empty-prompt) text embedding, computed once and reused.
    with torch.no_grad():
        tokens = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(tokens.input_ids.to(device))[0]

    return SDBundle(vae=vae, unet=unet, scheduler=scheduler,
                    text_embeddings=text_embeddings, device=device, dtype=dtype)
