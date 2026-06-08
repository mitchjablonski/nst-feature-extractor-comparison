"""OpenAI ADM (guided-diffusion) backend: general ImageNet pixel-space diffusion
features. Like the DDPM backend it reads UNet activations at a fixed noised
timestep and optimizes pixels directly (no VAE), but with ImageNet-general
features instead of LSUN-churches.

Requires the guided_diffusion package and the unconditional 256x256 checkpoint
(see README); both are fetched by scripts/setup_adm.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .base import Backend, HookManager

# guided_diffusion is git-cloned (not pip-installable); add it to the path.
_GD_PATH = Path(__file__).resolve().parents[3] / "third_party" / "guided-diffusion"
if _GD_PATH.is_dir() and str(_GD_PATH) not in sys.path:
    sys.path.insert(0, str(_GD_PATH))

# Hyperparameters of the public 256x256_diffusion_uncond.pt checkpoint.
_ADM_256_UNCOND = dict(
    num_channels=256,
    num_res_blocks=2,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    resblock_updown=True,
    use_scale_shift_norm=True,
    learn_sigma=True,
    class_cond=False,
    use_fp16=False,
    # Checkpoint ALL blocks (not just attention): our non-reentrant monkeypatch
    # makes it work with frozen params, and recomputing the big ResBlock
    # activations is what keeps peak VRAM low.
    use_checkpoint=True,
    diffusion_steps=1000,
    noise_schedule="linear",
)


class ADMBackend(Backend):
    space = "pixel"

    def __init__(self, config):
        super().__init__(config)
        import torch.utils.checkpoint as torch_ckpt

        import guided_diffusion.unet as gd_unet
        from guided_diffusion.script_util import (
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )

        # guided_diffusion's AttentionBlock hardcodes its custom CheckpointFunction
        # (unet.py: `checkpoint(..., True)`), whose backward differentiates w.r.t.
        # the frozen params and errors. Swap in PyTorch's non-reentrant checkpoint:
        # it recomputes activations (memory saving) AND tolerates frozen params.
        def _checkpoint(func, inputs, params, flag):
            if flag:
                return torch_ckpt.checkpoint(func, *inputs, use_reentrant=False)
            return func(*inputs)

        gd_unet.checkpoint = _checkpoint

        self.dtype = getattr(torch, config.dtype)
        args = model_and_diffusion_defaults()
        args.update(_ADM_256_UNCOND)
        args["image_size"] = config.adm_image_size
        model, diffusion = create_model_and_diffusion(**args)
        state = torch.load(config.adm_checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model.eval().requires_grad_(False)
        model.to(self.device)
        self.model = model
        self.alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=self.device).float()
        self.timestep = int(config.timestep)
        self.seed = int(config.seed)
        union = (*config.content_layers, *config.style_layers)
        self.hooks = HookManager(self.model, union)
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
        self.model(x_t, timesteps)
        return self.hooks.collect()

    def close(self) -> None:
        self.hooks.remove()
