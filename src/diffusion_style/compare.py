"""Comparison harness: run the same content+style pair through several feature
extractors and tile the results into one labeled contact sheet. The payoff of
the backend abstraction -- every lens side by side on identical inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .backends import BACKENDS
from .config import backend_config
from .sweep import contact_sheet
from .transfer import run_transfer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="diffusion-style-compare",
        description="Run several backends on one pair and build a contact sheet.",
    )
    p.add_argument("--content", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--backends", default="vgg,convnext,sd,ddpm",
                   help=f"comma-separated subset of: {','.join(BACKENDS)}")
    p.add_argument("--steps", type=int, help="override step count for all backends")
    p.add_argument("--display", type=int, default=384, help="contact-sheet cell size (px)")
    p.add_argument("--cols", type=int, default=0, help="grid columns (0 = one row)")
    args = p.parse_args(argv)

    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        p.error(f"unknown backend(s): {', '.join(unknown)} (choices: {', '.join(BACKENDS)})")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells = []
    for name in names:
        overrides = {"steps": args.steps} if args.steps else {}
        config = backend_config(name, **overrides)
        print(f"[{name}] running ...", flush=True)
        try:
            img = run_transfer(config, args.content, args.style)
        except Exception as exc:  # one bad backend shouldn't sink the sheet
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        img.save(outdir / f"{name}.png")
        cells.append((name, img.resize((args.display, args.display))))

    if not cells:
        print("no backends produced output")
        return
    cols = args.cols or len(cells)
    sheet_path = outdir / "compare.png"
    contact_sheet(cells, cols).save(sheet_path)
    print(f"saved {len(cells)} outputs + contact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
