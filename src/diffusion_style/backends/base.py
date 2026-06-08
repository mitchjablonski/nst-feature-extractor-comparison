"""Backend abstraction: a frozen feature extractor + its optimization space.

Every lens (VGG, SD-UNet, DDPM, ConvNeXt, DINOv3, ADM) implements this one
interface so a single optimization engine (transfer.run_transfer) can drive all
of them. That's what makes the comparison fair: the losses and optimizer are
identical; only the features differ.

The optimization variable `x` lives in the backend's `space`:
  - "pixel": x is an image tensor in [-1, 1]            (encode = identity)
  - "latent": x is a VAE latent                         (encode/decode via VAE)
"""

from __future__ import annotations

import torch

from .. import images


class HookManager:
    """Registers forward hooks on named submodules and collects their outputs.

    Block/stage outputs are captured; tuple outputs (e.g. diffusers down blocks)
    are reduced to their first element. Call reset() before each forward.
    """

    def __init__(self, module: torch.nn.Module, names):
        self.names = list(dict.fromkeys(names))
        self._feats: dict[str, torch.Tensor] = {}
        self._handles = [
            module.get_submodule(n).register_forward_hook(self._make(n)) for n in self.names
        ]

    def _make(self, name: str):
        def hook(_m, _i, out):
            self._feats[name] = out[0] if isinstance(out, tuple) else out
        return hook

    def reset(self) -> None:
        self._feats = {}

    def collect(self) -> dict[str, torch.Tensor]:
        return {n: self._feats[n] for n in self.names}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class Backend:
    """Base class for all feature-extractor backends. Defaults to pixel space."""

    space = "pixel"

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float32

    # --- optimization-space conversions (pixel-space defaults) ---
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image in [-1, 1] -> optimization variable x (fp32)."""
        return image.float()

    def to_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable x -> image tensor in [0, 1] (for the TV term)."""
        return (x + 1.0) / 2.0

    def clamp_(self, x: torch.Tensor) -> None:
        """In-place clamp x to its valid range during optimization. Pixel
        default keeps images in [-1, 1]; latent backends override to no-op."""
        x.clamp_(-1.0, 1.0)

    def to_pil(self, x: torch.Tensor):
        return images.tensor_to_pil(x.detach())

    # --- the lens ---
    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def close(self) -> None:
        """Release hooks/resources. Optional."""
