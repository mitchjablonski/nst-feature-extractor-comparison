"""Sweep tool: vary the feature-extraction timestep (and optionally style weight)
and lay the results out in a labeled contact sheet so operating points are
comparable at a glance.

Timestep strongly controls which UNet features dominate (textural vs semantic),
so sweeping it is the diffusion analog of choosing which VGG layers to use.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from .config import Config
from .transfer import run_transfer


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def contact_sheet(cells: list[tuple[str, Image.Image]], cols: int) -> Image.Image:
    """Tile labeled images into a grid (label band above each)."""
    if not cells:
        raise ValueError("no cells to tile")
    band = 18
    w, h = cells[0][1].size
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + band)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(cells):
        r, c = divmod(i, cols)
        x, y = c * w, r * (h + band)
        draw.text((x + 2, y + 4), label, fill="black")
        sheet.paste(img, (x, y + band))
    return sheet


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="diffusion-style-sweep",
        description="Sweep timestep / style weight and build a contact sheet.",
    )
    p.add_argument("--content", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--timesteps", default="101,261,461,661",
                   help="comma-separated timesteps (0..999)")
    p.add_argument("--style-weights", default="",
                   help="optional comma-separated style weights; default uses Config's")
    p.add_argument("--steps", type=int, default=Config().steps)
    p.add_argument("--resolution", type=int, default=Config().resolution)
    p.add_argument("--cols", type=int, default=0, help="grid columns (0 = #timesteps)")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    timesteps = _parse_int_list(args.timesteps)
    style_weights = _parse_float_list(args.style_weights) or [Config().style_weight]

    cells: list[tuple[str, Image.Image]] = []
    for sw in style_weights:
        for t in timesteps:
            config = replace(
                Config(), timestep=t, style_weight=sw,
                steps=args.steps, resolution=args.resolution,
            )
            print(f"running t={t} style_weight={sw:g} ...")
            img = run_transfer(config, args.content, args.style)
            name = f"t{t}_sw{sw:g}.png"
            img.save(outdir / name)
            cells.append((f"t={t} sw={sw:g}", img))

    cols = args.cols or len(timesteps)
    sheet = contact_sheet(cells, cols)
    sheet_path = outdir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"saved {len(cells)} images + contact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
