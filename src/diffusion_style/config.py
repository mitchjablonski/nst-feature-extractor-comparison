"""Central configuration for a style-transfer run.

Every knob the CLI, sweep tool, and notebook can touch lives here, so all
entry points drive the same surface (REQ-2: configurable from one place).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Default to CUDA when present, otherwise CPU (slow but functional — the
# feed-forward backends are usable). Override per run with --device
# (e.g. "mps" on Apple silicon).
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The original runwayml/stable-diffusion-v1-5 repo was removed; this is the
# community-maintained mirror with identical weights.
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# We hook block-level outputs of UNet2DConditionModel. Block outputs are the
# boundaries between gradient-checkpoint segments, so they are retained during
# the checkpointed forward (deep internal activations would be recomputed and
# unavailable to a hook). Names resolve via unet.get_submodule(name).
#
# Style: multi-scale, spanning encoder + decoder for texture at several scales.
DEFAULT_STYLE_LAYERS = (
    "down_blocks.0",
    "down_blocks.1",
    "down_blocks.2",
    "up_blocks.1",
    "up_blocks.2",
)
# Content: a single mid-deep decoder block preserves layout without locking in
# fine texture (the diffusion analog of VGG conv4_2).
DEFAULT_CONTENT_LAYERS = ("up_blocks.1",)

# --- Pixel-space DDPM backend (no VAE; optimize pixels directly) ---
# google/ddpm-church-256 is a pixel UNet2DModel with blocks down to full 256px
# resolution, so fine pixel-level texture is available (latent space lacked it).
# Block output resolutions: down0=128, down1=64, down2=32, down3=16, down4=8.
DDPM_MODEL_ID = "google/ddpm-church-256"
# VGG-like multi-scale spread across the encoder path (128 -> 8 px).
DDPM_STYLE_LAYERS = (
    "down_blocks.0",
    "down_blocks.1",
    "down_blocks.2",
    "down_blocks.3",
    "down_blocks.4",
)
# Mid-deep block for content (analog of VGG conv4_2).
DDPM_CONTENT_LAYERS = ("down_blocks.3",)

# --- Feed-forward CNN backends (no diffusion noise) ---
# Names index into the .features Sequential directly (the backend hooks that).
# VGG-19 conv indices: conv1_1..conv5_1 for style, conv4_2 for content.
VGG_STYLE_LAYERS = ("0", "5", "10", "19", "28")
VGG_CONTENT_LAYERS = ("21",)
# ConvNeXt-Large: hook the RAW conv outputs BEFORE each LayerNorm. LayerNorm
# whitens the activation magnitudes that Gram-style transfer needs, so the
# post-norm block outputs barely stylize; these pre-norm points work (verified).
# "0.0" = stem conv; "<stage>.0.block.0" = a stage's first depthwise conv.
CONVNEXT_STYLE_LAYERS = ("0.0", "1.0.block.0", "3.0.block.0", "5.0.block.0")
CONVNEXT_CONTENT_LAYERS = ("3.0.block.0",)
# DINOv3-ConvNeXt backbone feature_maps, named stage0..stage3 by the adapter.
DINOV3_STYLE_LAYERS = ("stage0", "stage1", "stage2")
DINOV3_CONTENT_LAYERS = ("stage2",)
# DINOv2 ViT transformer-layer indices into hidden_states. SHALLOW blocks carry
# transferable style; deep blocks are too semantic and barely stylize (verified).
DINOV2_STYLE_LAYERS = ("layer2", "layer4", "layer6", "layer8")
DINOV2_CONTENT_LAYERS = ("layer8",)
# OpenAI ADM (guided-diffusion) UNet, 18 input_blocks. Multi-scale encoder
# spread (256/128/64/32/16 px); content from a mid-deep 64px block (~conv4_2).
ADM_STYLE_LAYERS = (
    "input_blocks.2", "input_blocks.5", "input_blocks.8",
    "input_blocks.11", "input_blocks.14",
)
ADM_CONTENT_LAYERS = ("input_blocks.8",)


@dataclass(frozen=True)
class Config:
    backend: str = "sd"  # sd | ddpm | vgg | convnext | dinov3 | adm
    model_id: str = DEFAULT_MODEL_ID
    ddpm_model_id: str = DDPM_MODEL_ID
    dinov3_model_id: str = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
    dinov2_model_id: str = "facebook/dinov2-large"
    adm_checkpoint: str = "models/256x256_diffusion_uncond.pt"
    adm_image_size: int = 256

    # Fixed denoising timestep at which UNet features are read (0..999).
    # ~261/1000 follows DIFT: low enough that the image still dominates the
    # noised latent, high enough that semantic features have formed.
    timestep: int = 261

    content_layers: tuple[str, ...] = DEFAULT_CONTENT_LAYERS
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS

    content_weight: float = 1.0
    # Gram MSE is normalized by C*H*W, so it lands around 1e-4; 1e6 is what makes
    # the style term actually drive the latent (verified empirically). Lower this
    # to preserve more content, raise it for stronger stylization.
    style_weight: float = 1.0e6
    # Total-variation weight: penalizes pixel-space grain from the VAE decode.
    # 0 disables it; raise for a smoother (more CNN-like) result, but too high
    # washes out brushwork. Tuned empirically via a tv_weight sweep.
    tv_weight: float = 0.0

    resolution: int = 512  # px; latent is resolution/8 square. Peak VRAM ~2.3GB at 384, so 512 fits.

    # Lower lr + more steps = gentler convergence, less residual high-frequency
    # noise (matches the well-tuned TF/Gatys tutorial recipe). This is an
    # optimizer setting, so it improves every pixel backend uniformly/fairly.
    steps: int = 300
    lr: float = 0.02
    optimizer: str = "adam"  # "adam" | "lbfgs"

    seed: int = 0  # seeds the fixed feature-extraction noise (determinism)

    device: str = DEFAULT_DEVICE
    dtype: str = "float16"  # UNet/VAE compute dtype; the optimized latent stays fp32


# Per-backend presets: each lens has different layer names, feature scales, and
# good weight ranges. SD uses the dataclass defaults; the rest override here.
# style_weight values are starting points to be retuned per backend via sweeps.
_PRESETS: dict[str, dict] = {
    # SD latent optimization DIVERGES into speckle with more steps (no smoothness
    # prior on the latent; VAE decode amplifies it) -> pin steps low (~250).
    "sd": dict(steps=250),
    "ddpm": dict(
        backend="ddpm", timestep=100, steps=500,
        content_layers=DDPM_CONTENT_LAYERS, style_layers=DDPM_STYLE_LAYERS,
        style_weight=1.0e6,
        # TV ~30 clears the pixel speckle WITHOUT fighting content (verified:
        # content loss stayed flat ~1.3 across tv 0->30). The pixel-space win.
        tv_weight=30.0, resolution=256, dtype="float32",
    ),
    # Feed-forward backends use LBFGS (cheap forward, and LBFGS is what makes
    # NST converge to strong stylization — the Gatys/PyTorch-tutorial recipe).
    # No TV needed: LBFGS doesn't produce the Adam grain we were over-correcting.
    "vgg": dict(
        backend="vgg", optimizer="lbfgs", steps=500,
        content_layers=VGG_CONTENT_LAYERS, style_layers=VGG_STYLE_LAYERS,
        # sw3e6 (sweep winner) gives more pronounced brushwork than 1e6.
        style_weight=3.0e6, tv_weight=0.0, resolution=512, dtype="float32",
    ),
    "convnext": dict(
        backend="convnext", optimizer="lbfgs", steps=500,
        content_layers=CONVNEXT_CONTENT_LAYERS, style_layers=CONVNEXT_STYLE_LAYERS,
        # pre-LayerNorm hooks + ~1e8 give a real transfer (post-norm barely did).
        style_weight=1.0e8, tv_weight=0.0, resolution=512, dtype="float32",
    ),
    "dinov3": dict(
        backend="dinov3", optimizer="lbfgs", steps=500,
        content_layers=DINOV3_CONTENT_LAYERS, style_layers=DINOV3_STYLE_LAYERS,
        # sw1e8 gives marginally more; DINOv3 stays weak (SSL appearance-invariance).
        style_weight=1.0e8, tv_weight=0.0, resolution=256, dtype="float32",
    ),
    "dinov2": dict(
        backend="dinov2", optimizer="lbfgs", steps=500,
        content_layers=DINOV2_CONTENT_LAYERS, style_layers=DINOV2_STYLE_LAYERS,
        # 224 = multiple of the patch size (14). Shallow layers + ~1e8 stylize.
        style_weight=1.0e8, tv_weight=0.0, resolution=224, dtype="float32",
    ),
    "adm": dict(
        backend="adm", timestep=100, steps=500,
        content_layers=ADM_CONTENT_LAYERS, style_layers=ADM_STYLE_LAYERS,
        # fp32 + FULL non-reentrant activation checkpointing keeps peak VRAM low
        # (~a few GB) on 8GB without the guided_diffusion fp16 dtype-mixing pain.
        style_weight=1.0e6, tv_weight=30.0, resolution=256, dtype="float32",
    ),
}


def backend_config(name: str = "sd", **overrides) -> Config:
    """Build a Config for a given backend from its preset, plus any overrides."""
    if name not in _PRESETS:
        raise ValueError(f"unknown backend preset: {name!r} (have {list(_PRESETS)})")
    base = dict(_PRESETS[name])
    base.update(overrides)
    return Config(**base)


def ddpm_config(**overrides) -> Config:
    """Back-compat shim for the pixel-space DDPM preset."""
    return backend_config("ddpm", **overrides)
