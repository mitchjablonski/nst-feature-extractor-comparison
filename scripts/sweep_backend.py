"""Per-backend hyperparameter sweep harness. Runs a backend's predefined grid
(optimizer / layers / style_weight / tv / timestep) on a fixed tuning pair and
writes a labeled contact sheet + per-cell PNGs for visual selection.

Usage: uv run --extra adm python scripts/sweep_backend.py <backend> [content] [style]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image

from diffusion_style.config import backend_config
from diffusion_style.sweep import contact_sheet
from diffusion_style.transfer import run_transfer

# Ensure every config is compared at convergence (TF uses ~1000 Adam steps;
# LBFGS settles by a few hundred). Applied unless a grid entry overrides steps.
STEPS_BY_OPT = {"adam": 1000, "lbfgs": 500}

# Each entry: (label, config-override dict). Axes chosen from what actually moved
# results in debugging: optimizer, hook point (layers), style_weight, tv, timestep.
GRIDS: dict[str, list[tuple[str, dict]]] = {
    "vgg": [
        ("lbfgs_sw1e6_c42", dict(optimizer="lbfgs", style_weight=1e6, content_layers=("21",))),
        ("lbfgs_sw3e6_c42", dict(optimizer="lbfgs", style_weight=3e6, content_layers=("21",))),
        ("lbfgs_sw1e7_c42", dict(optimizer="lbfgs", style_weight=1e7, content_layers=("21",))),
        ("lbfgs_sw1e6_c52", dict(optimizer="lbfgs", style_weight=1e6, content_layers=("30",))),
        ("lbfgs_sw3e6_c52", dict(optimizer="lbfgs", style_weight=3e6, content_layers=("30",))),
        ("lbfgs_sw3e6_tv1e3", dict(optimizer="lbfgs", style_weight=3e6, tv_weight=1e3)),
        ("ADAM_sw1e6", dict(optimizer="adam", style_weight=1e6)),  # documents the gap
    ],
    "convnext": [  # preset = pre-norm layers
        ("prenorm_sw1e7", dict(style_weight=1e7)),
        ("prenorm_sw1e8", dict(style_weight=1e8)),
        ("prenorm_sw1e9", dict(style_weight=1e9)),
        ("prenorm_sw1e8_tv10", dict(style_weight=1e8, tv_weight=10)),
        ("POSTnorm_sw1e8", dict(style_weight=1e8, style_layers=("1", "3", "5", "7"),
                                content_layers=("5",))),  # documents post-norm fails
    ],
    "dinov2": [  # preset = shallow layers
        ("shallow_sw1e7", dict(style_weight=1e7)),
        ("shallow_sw1e8", dict(style_weight=1e8)),
        ("shallow_sw1e9", dict(style_weight=1e9)),
        ("mid_sw1e8", dict(style_weight=1e8, style_layers=("layer6", "layer9", "layer12", "layer15"),
                           content_layers=("layer12",))),
        ("DEEP_sw1e8", dict(style_weight=1e8, style_layers=("layer12", "layer16", "layer20"),
                            content_layers=("layer16",))),  # documents deep fails
    ],
    "dinov3": [  # post-norm feature_maps (pre-norm tested separately, only marginal)
        ("sw1e7", dict(style_weight=1e7)),
        ("sw1e8", dict(style_weight=1e8)),
    ],
    "ddpm": [
        ("t50_sw1e6_tv30", dict(timestep=50, style_weight=1e6, tv_weight=30)),
        ("t100_sw1e6_tv30", dict(timestep=100, style_weight=1e6, tv_weight=30)),
        ("t200_sw1e6_tv30", dict(timestep=200, style_weight=1e6, tv_weight=30)),
        ("t100_sw1e7_tv30", dict(timestep=100, style_weight=1e7, tv_weight=30)),
        ("t100_sw1e6_tv10", dict(timestep=100, style_weight=1e6, tv_weight=10)),
    ],
    "adm": [
        ("t50_sw1e6", dict(timestep=50, style_weight=1e6, tv_weight=30)),
        ("t100_sw1e6", dict(timestep=100, style_weight=1e6, tv_weight=30)),
        ("t200_sw1e6", dict(timestep=200, style_weight=1e6, tv_weight=30)),
        ("t100_sw1e7", dict(timestep=100, style_weight=1e7, tv_weight=30)),
    ],
    "sd": [
        ("t100_sw1e6", dict(timestep=100, style_weight=1e6)),
        ("t261_sw1e6", dict(timestep=261, style_weight=1e6)),
        ("t461_sw1e6", dict(timestep=461, style_weight=1e6)),
        ("t261_sw1e7", dict(timestep=261, style_weight=1e7)),
    ],
}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    name = argv[0]
    content = argv[1] if len(argv) > 1 else "assets/barn.jpg"
    style = argv[2] if len(argv) > 2 else "assets/vangogh.jpg"
    outdir = Path("outputs/sweep") / name
    outdir.mkdir(parents=True, exist_ok=True)
    cells = []
    for label, ov in GRIDS[name]:
        cfg = backend_config(name, **ov)
        if "steps" not in ov:
            cfg = replace(cfg, steps=STEPS_BY_OPT.get(cfg.optimizer, 1000))
        try:
            img = run_transfer(cfg, content, style)
        except Exception as exc:
            print(f"[{name}:{label}] FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        img.save(outdir / f"{label}.png")
        cells.append((label, img.resize((300, 300))))
        print(f"[{name}:{label}] done", flush=True)
    if cells:
        contact_sheet(cells, cols=min(4, len(cells))).save(outdir / "sheet.png")
        print(f"[{name}] SHEET -> {outdir}/sheet.png", flush=True)


if __name__ == "__main__":
    main()
