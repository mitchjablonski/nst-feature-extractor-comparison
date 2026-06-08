"""The style-transfer optimization engine, backend-agnostic.

One loop drives every feature-extractor backend through the Backend interface:
encode content+style, precompute detached content + Gram-style targets, then
optimize x (initialized from content) with content + style + optional TV losses.
Identical losses/optimizer across backends is what makes the comparison fair.
The model is always frozen; only x has requires_grad.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image

from . import images, losses
from .backends import get_backend
from .config import Config


@dataclass
class StepLog:
    step: int
    total: float
    content: float
    style: float
    tv: float = 0.0


ProgressFn = Callable[[StepLog], None]


def run_transfer(
    config: Config,
    content_path: str,
    style_path: str,
    progress: ProgressFn | None = None,
) -> Image.Image:
    torch.manual_seed(config.seed)
    backend = get_backend(config)

    content_img = images.load_image(content_path, config.resolution, backend.device, backend.dtype)
    style_img = images.load_image(style_path, config.resolution, backend.device, backend.dtype)
    content_x = backend.encode(content_img)
    style_x = backend.encode(style_img)

    # Precompute detached targets once.
    with torch.no_grad():
        cfeats = backend.features(content_x)
        content_targets = {n: cfeats[n].float().detach() for n in config.content_layers}
        sfeats = backend.features(style_x)
        target_grams = {n: losses.gram_matrix(sfeats[n]).detach() for n in config.style_layers}

    # Optimize x (image or latent, per backend.space), initialized from content.
    # .contiguous() is required for LBFGS's flat-gradient gather.
    x = content_x.clone().detach().to(torch.float32).contiguous().requires_grad_(True)

    def closure() -> torch.Tensor:
        with torch.no_grad():
            backend.clamp_(x)  # keep x in its valid range (pixels in [-1, 1])
        optimizer.zero_grad()
        feats = backend.features(x)
        c = losses.content_loss(feats, content_targets, config.content_layers)
        s = losses.style_loss(feats, target_grams, config.style_layers)
        loss = config.content_weight * c + config.style_weight * s
        tv_val = 0.0
        if config.tv_weight > 0.0:
            tv = losses.total_variation(backend.to_pixels(x).float())
            loss = loss + config.tv_weight * tv
            tv_val = tv.item()
        loss.backward()
        if progress is not None:
            progress(StepLog(closure.calls, loss.item(), c.item(), s.item(), tv_val))
        closure.calls += 1
        return loss

    closure.calls = 0

    if config.optimizer == "lbfgs":
        # Second-order; converges to far stronger stylization than Adam (this is
        # what the Gatys/PyTorch tutorials use). config.steps is LBFGS max_iter;
        # each iter does several closure evals via line search.
        optimizer = torch.optim.LBFGS([x], max_iter=config.steps, line_search_fn="strong_wolfe")
        optimizer.step(closure)
    else:
        optimizer = torch.optim.Adam([x], lr=config.lr)
        for _ in range(config.steps):
            closure()
            optimizer.step()

    with torch.no_grad():
        backend.clamp_(x)
    result = backend.to_pil(x.detach())
    backend.close()
    return result
