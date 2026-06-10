"""Gatys content + Gram-matrix style losses, backend-agnostic.

This is the Gatys method preserved verbatim; the features may come from any
backend (VGG, diffusion UNets, ConvNeXt, DINO). Losses run in fp32 for
stability even when a backend computes features in fp16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
    """Channel correlation matrix, normalized by feature size.

    feature: [B, C, H, W] -> [B, C, C]. Normalizing by C*H*W keeps the loss
    scale comparable across layers of different sizes.
    """
    b, c, h, w = feature.shape
    flat = feature.reshape(b, c, h * w).float()
    gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram / (c * h * w)


def content_loss(feats: dict, targets: dict, layers) -> torch.Tensor:
    """Sum of feature MSE over the content layers."""
    return sum(F.mse_loss(feats[name].float(), targets[name]) for name in layers)


def style_loss(feats: dict, target_grams: dict, layers) -> torch.Tensor:
    """Sum of Gram-matrix MSE over the style layers."""
    return sum(F.mse_loss(gram_matrix(feats[name]), target_grams[name]) for name in layers)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference between neighboring pixels (a smoothness penalty).

    image: [B, 3, H, W]. Penalizes high-frequency grain directly in pixel space,
    independent of the style layers, so we can keep strong brushwork (fine
    up_blocks) while suppressing the roughness the VAE decoder introduces.
    """
    dh = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    dw = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return dh + dw
