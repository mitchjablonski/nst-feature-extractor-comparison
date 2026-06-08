"""CLI entrypoint: one reproducible command for a style-transfer run (REQ-5)."""

from __future__ import annotations

import argparse

from .backends import BACKENDS
from .config import backend_config
from .transfer import StepLog, run_transfer

# CLI flag -> Config field. All default to None so an unset flag means
# "use the backend preset"; only provided flags override the preset.
_TUNABLES = (
    "steps", "content_weight", "style_weight", "tv_weight",
    "timestep", "resolution", "lr", "optimizer", "seed",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diffusion-style",
        description="Gatys-style transfer with a choice of frozen feature extractor.",
    )
    p.add_argument("--content", required=True, help="content image path")
    p.add_argument("--style", required=True, help="style image path")
    p.add_argument("--out", required=True, help="output image path")
    p.add_argument("--backend", choices=BACKENDS, default="sd",
                   help="feature extractor: " + ", ".join(BACKENDS))
    # Tunables default to None -> fall back to the backend's preset.
    p.add_argument("--steps", type=int)
    p.add_argument("--content-weight", type=float)
    p.add_argument("--style-weight", type=float)
    p.add_argument("--tv-weight", type=float)
    p.add_argument("--timestep", type=int, help="0..999 (diffusion backends only)")
    p.add_argument("--resolution", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--optimizer", choices=("adam", "lbfgs"))
    p.add_argument("--seed", type=int)
    p.add_argument("--ddpm-model", help="override the pixel DDPM model id")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    overrides = {f: getattr(args, f) for f in _TUNABLES if getattr(args, f) is not None}
    if args.ddpm_model is not None:
        overrides["ddpm_model_id"] = args.ddpm_model
    config = backend_config(args.backend, **overrides)

    def progress(log: StepLog) -> None:
        if log.step % 10 == 0 or log.step == config.steps - 1:
            print(
                f"step {log.step:4d}  total {log.total:.4e}  content {log.content:.4e}  "
                f"style {log.style:.4e}  tv {log.tv:.4e}"
            )

    image = run_transfer(config, args.content, args.style, progress=progress)
    image.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
