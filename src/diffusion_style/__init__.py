"""Gatys-style neural style transfer with interchangeable frozen feature extractors.

The classic Gatys method uses a frozen VGG as a fixed feature extractor and
optimizes an image so its deep features match a content image (raw MSE) and its
style matches (MSE of Gram matrices). This package keeps that engine fixed and
makes the extractor pluggable: VGG-19, SD1.5 UNet (latent), pixel DDPM,
ConvNeXt, DINOv2/v3, and OpenAI ADM — all driven by the same losses and
optimizer so the lenses can be compared fairly.
"""

__version__ = "0.1.0"
