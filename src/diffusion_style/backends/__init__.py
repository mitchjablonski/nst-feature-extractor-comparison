"""Backend registry. Lazy imports so a missing optional dep (e.g. ADM's
guided_diffusion) only errors when that backend is actually requested.
"""

from __future__ import annotations

from .base import Backend, HookManager

# Names usable as --backend / in compare runs.
BACKENDS = ("vgg", "sd", "ddpm", "convnext", "dinov2", "dinov3", "adm")


def get_backend(config) -> Backend:
    name = config.backend
    if name == "sd":
        from .sd import SDBackend
        return SDBackend(config)
    if name == "ddpm":
        from .ddpm import DDPMBackend
        return DDPMBackend(config)
    if name == "vgg":
        from .vgg import VGGBackend
        return VGGBackend(config)
    if name == "convnext":
        from .convnext import ConvNeXtBackend
        return ConvNeXtBackend(config)
    if name == "dinov2":
        from .dinov2 import DINOv2Backend
        return DINOv2Backend(config)
    if name == "dinov3":
        from .dinov3 import DINOv3Backend
        return DINOv3Backend(config)
    if name == "adm":
        from .adm import ADMBackend
        return ADMBackend(config)
    raise ValueError(f"unknown backend: {name!r} (choices: {', '.join(BACKENDS)})")


__all__ = ["Backend", "HookManager", "BACKENDS", "get_backend"]
