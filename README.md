# diffusion-style

A **comparison harness** for Gatys-style neural style transfer with
interchangeable frozen feature extractors. Same optimization engine, same
losses — swap only the "lens" and compare.

The classic [Gatys method](https://arxiv.org/abs/1508.06576) treats a frozen
network as a fixed lens: you optimize an image so its deep features match a
**content** image (feature MSE) and its **style** matches (MSE of feature
**Gram matrices**). This repo keeps that engine fixed and makes the *extractor*
pluggable, so you can ask: do diffusion features beat VGG? Do modern
self-supervised CNNs beat both?

## Backends (the lenses)

Selected with `--backend`. Each plugs into one shared optimization loop.

| Backend | Lens | Space | Notes |
|---------|------|-------|-------|
| `vgg` | VGG-19 (ImageNet) | pixel | classic baseline; clean & painterly |
| `sd` | Stable Diffusion 1.5 UNet | latent | rough — VAE/latent bottleneck; diverges with steps |
| `ddpm` | Pixel DDPM (LSUN-church 256) | pixel | painterly (Adam + TV); domain-narrow (churches) |
| `convnext` | ConvNeXt-Large (ImageNet) | pixel | stylizes with **pre-LayerNorm** hooks, but milder than VGG |
| `dinov2` | DINOv2-Large (ViT) | pixel | self-supervised; partial (needs **shallow** layers) |
| `dinov3` | DINOv3-ConvNeXt-Large | pixel | self-supervised; weak (appearance-invariant); **gated HF repo** |
| `adm` | OpenAI ADM / guided-diffusion (ImageNet 256) | pixel | strongest diffusion lens; needs `scripts/setup_adm.sh` |

See `examples/` for results across three styles (Van Gogh, Gris cubist, Kandinsky).

### What we found

Three axes turned out to matter — and two of them surprised us:

- **The optimizer is decisive: LBFGS ≫ Adam.** Adam silently *under-converges*
  on Gatys NST, producing muted, photo-like results; LBFGS (second-order, line
  search) is what the Gatys/PyTorch tutorials use and what lets the style fully
  develop. This single setting was responsible for most of our early "bad"
  results. Feed-forward backends use LBFGS; diffusion backends use Adam (LBFGS's
  many per-iteration UNet evals are too slow).
- **Hook point matters — but so does architecture.** ConvNeXt produced *nothing*
  until we hooked the **pre-LayerNorm** conv outputs: LayerNorm normalizes away
  the exact magnitude statistics a Gram matrix needs, so post-norm features don't
  stylize at all. Pre-norm hooks unlock it — but the result is still **milder
  than VGG**, because ConvNeXt's `4×4` stride-4 patchify stem has no
  full-resolution fine-texture layers (VGG's `conv1_1/1_2` do). So both factors
  are real: the hook point was blocking it entirely; the stem caps how strong it
  can get.
- **Supervised features carry style; self-supervised (DINO) features don't.**
  VGG and ConvNeXt (supervised ImageNet) transfer style well. DINO is trained to
  be *invariant* to appearance/color — the opposite of what Gram-NST wants — so
  DINOv2 only partly works (and only from **shallow** ViT layers; deep layers are
  too semantic), and DINOv3 stays weak even with pre-norm hooks. (The method that
  *does* make DINO work — Splice — trains a generator instead of optimizing
  pixels; it's a different pipeline, not a lens swap.)
- **SD-latent *diverges* with more steps.** Unlike the pixel backends (which
  converge), optimizing the SD latent over-fits into speckle as steps increase
  (no smoothness prior on the latent; the VAE decode amplifies it) — so SD runs
  fewer steps (~250) and is still the roughest lens. A clean illustration of the
  latent/VAE bottleneck that motivated the pixel-space backends.
- **ADM (ImageNet pixel-diffusion) is the strongest diffusion lens**, rivaling
  VGG; the LSUN-church DDPM is good but domain-narrow.

**Takeaway:** for Gatys-style direct optimization, **VGG and pixel-space
diffusion (ADM/DDPM) are the strongest, cleanest lenses**. **ConvNeXt** stylizes
(once hooked pre-norm) but stays **milder** than VGG. **SD-latent is inherently
rough**, and **self-supervised (DINO) features are the genuine weak case** —
invariance is the point of SSL, and it's exactly what style transfer doesn't
want. Net: supervised CNN + pixel-diffusion features win; the modern/SSL nets
range from milder (ConvNeXt) to weak (DINO).

### Tuning & caveats
Per-backend hyperparameters (optimizer, hook layers, style weight, TV, timestep)
were swept **at convergence** (Adam 1000 / LBFGS 500 steps) on one pair
(barn + Van Gogh) and validated to **generalize across all three styles**. The
cubist results stylize gently because the *Gris style image* is muted/low-contrast
— a property of the input, not the method. DINOv3 requires gated Hugging Face
access; the `adm` backend needs the one-time `scripts/setup_adm.sh`.

## How it works

1. Pick a backend; it loads a frozen model and exposes multi-scale feature maps.
2. Encode content & style into the backend's space (pixels, or a VAE latent).
3. Read features (for diffusion backends, at a fixed noised timestep via hooks).
4. Optimize `x` (init from content) so content features match (MSE) and style
   Gram matrices match (MSE), plus an optional TV smoothness term. One shared
   engine, with LBFGS for the feed-forward backends and Adam for the (expensive)
   diffusion backends.
5. Decode `x` to the output image.

## Setup

```bash
uv sync                       # CUDA torch (cu124) + diffusers + transformers + torchvision

# Optional, only for specific backends:
bash scripts/setup_adm.sh     # ADM: blobfile + git-clone guided_diffusion + 2GB checkpoint
uv run huggingface-cli login  # DINOv3: gated repo — accept its license first
```

First run of each backend downloads its weights from the Hugging Face Hub.

**ADM note:** `guided_diffusion` is an optional extra, so ADM runs must pass the
flag, e.g. `uv run --extra adm diffusion-style --backend adm ...`.

## Usage

```bash
# Single transfer with a chosen backend
uv run diffusion-style --backend ddpm \
  --content assets/tetons.jpg --style assets/vangogh.jpg --out out.png

# Compare several backends on one pair -> labeled contact sheet
uv run diffusion-style-compare \
  --content assets/tetons.jpg --style assets/vangogh.jpg \
  --outdir outputs/compare --backends vgg,convnext,sd,ddpm

# Sweep the feature timestep (diffusion backends)
uv run diffusion-style-sweep \
  --content assets/tetons.jpg --style assets/vangogh.jpg \
  --outdir sweeps/ --timesteps 50,100,261,461
```

Sample pairs in `assets/`: `tetons.jpg`+`vangogh.jpg`, `portrait.jpg`+`gris.jpg`,
`cityscape.jpg`+`kandinsky.jpg`. Curated results per style live in `examples/`
(the CLI/compare/sweep tools write scratch to `outputs/`, which is gitignored).

Per-backend defaults live in `config.py` (`backend_config`/`_PRESETS`). Override
any knob via flags: `--timestep`, `--content-weight`, `--style-weight`,
`--tv-weight`, `--resolution`, `--steps`, `--lr`.

## Layout

| File | Role |
|------|------|
| `config.py` | every knob + per-backend presets (`backend_config`) |
| `backends/base.py` | `Backend` interface + shared `HookManager` |
| `backends/{vgg,sd,ddpm,convnext,dinov3,adm}.py` | the lenses |
| `images.py` | image <-> tensor/latent helpers (VAE + scaling) |
| `losses.py` | content + Gram style + total-variation losses |
| `transfer.py` | the single backend-agnostic optimization engine |
| `cli.py` / `sweep.py` / `compare.py` | entry points |
| `model.py` | frozen SD1.5 component loader (used by the `sd` backend) |

## Built with

This repo was built collaboratively using **DeepPairing**, an MCP server that
turns coding into a reviewed pair-programming loop — every plan, decision,
research finding, and code change is surfaced as an artifact for human review
before it lands. The SD → DDPM → TV → multi-backend evolution here was driven
through that review loop. See [DeepPairing](https://github.com/mitchjablonski/deepPairing).

## License

Code: **MIT** (see `LICENSE`). Input images keep their own licenses — see
`CREDITS.md` (artworks + the Tübingen photo are public domain; the portrait is
CC BY-SA 4.0, © Petar Milošević). Model weights are downloaded from their
original sources on first run and are not redistributed here.
