# Examples — feature-extractor comparison across three styles

Same Gatys engine (content MSE + Gram style + TV), swapping only the frozen
feature extractor. **Three content+style pairs**, each run through all seven
lenses — to show the ranking is a property of the *extractor*, not the style.

| Folder | Content + Style | Style character |
|--------|-----------------|-----------------|
| `vangogh/`  | red barn (Highsmith, PD) + Van Gogh *Farmhouse in Provence* | soft, swirly texture |
| `cubist/`   | portrait + Juan Gris *Portrait of Picasso*    | hard cubist planes |
| `abstract/` | Tübingen cityscape + Kandinsky *Composition VII* | chaotic abstraction |

Each folder has one tuned result per backend plus `comparison.png` — 7 lenses in
a 4×2 grid (row 1: vgg · ddpm · adm · convnext; row 2: sd · dinov2 · dinov3).
Inputs in `inputs/`.

## What each lens gives you (character holds across all three styles)

| Backend | Lens | Result |
|---------|------|--------|
| `vgg` | VGG-19 (ImageNet) | ✅ clean, classic painterly — the baseline |
| `adm` | OpenAI ADM (ImageNet diffusion) | ✅ strongest diffusion lens — clean & painterly |
| `ddpm` | Pixel DDPM (LSUN-church) | ✅ smooth painterly (TV clears speckle); domain-narrow |
| `convnext` | ConvNeXt-Large | ⚠️ stylizes (pre-norm hooks) but **milder than VGG** — coarse patchify stem |
| `dinov2` | DINOv2 (ViT) | ⚠️ partial — real transfer only from **shallow** layers |
| `sd` | Stable Diffusion 1.5 (latent) | ⚠️ rough — latent/VAE bottleneck; diverges with steps |
| `dinov3` | DINOv3-ConvNeXt | ⚠️ weak — SSL appearance-invariance (not fixable) |

**Takeaway:** **VGG and pixel-space diffusion (ADM/DDPM) are the strongest,
cleanest lenses.** **ConvNeXt** stylizes (once hooked pre-norm) but stays
**milder** than VGG; **SD-latent is inherently rough**; **self-supervised (DINO)
features are the genuine weak case**. The per-backend character holds across all
three styles — it's a property of the *extractor*, not the style. (Cubist
stylizes gently across *every* lens because the Gris style image is
muted/low-contrast — an input property, not a backend difference.)

### Why supervised wins and self-supervised (DINO) doesn't
- **Supervised CNNs carry style.** VGG works out of the box. **ConvNeXt** also
  stylizes — but only once you hook its **pre-LayerNorm** conv outputs (LayerNorm
  whitens the magnitude statistics a Gram matrix needs, so post-norm produces
  nothing). Even then it's **milder than VGG**, because its `4×4` stride-4
  patchify stem has no full-resolution fine-texture layers — so the architecture,
  not just the hook point, caps it.
- **Self-supervised DINO features are trained to be appearance-invariant** — the
  opposite of what Gram-NST wants. DINOv2 only partly works, and only from
  **shallow** ViT layers (deep layers are too semantic); DINOv3 stays weak even
  with pre-norm hooks (we also tried a color-moment loss and CLS/Splice-style
  matching — no luck under direct pixel optimization).
- The method that *does* make DINO work — **Splice** (CLS appearance + key
  self-similarity) — trains a *generator* rather than optimizing pixels, so it's a
  different pipeline, not a lens swap into this engine.

## Inputs & licensing
`inputs/`: barn + Van Gogh, portrait + Gris, cityscape + Kandinsky. Artworks and
the Tübingen photo are public domain; the portrait photo is CC-BY-SA (Wikimedia
Commons featured picture).

Reproduce any cell, e.g.:
`uv run --extra adm diffusion-style --backend adm --content inputs/cityscape.jpg --style inputs/kandinsky.jpg --out out.png`
