"""Gatys-style neural style transfer driven by a frozen Stable Diffusion 1.5 UNet.

The classic Gatys method uses a frozen VGG as a fixed feature extractor and
optimizes an image so its deep features match a content image (raw MSE) and its
style matches (MSE of Gram matrices). Here the frozen feature extractor is the
SD1.5 UNet evaluated at a fixed denoising timestep, and we optimize the VAE
latent rather than raw pixels, decoding to an image at the end.
"""

__version__ = "0.1.0"
