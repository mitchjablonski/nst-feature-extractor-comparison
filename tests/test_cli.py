"""CLI surface tests: parsing and validation only (no transfers run)."""

import pytest

from diffusion_style import compare
from diffusion_style.cli import _TUNABLES, build_parser

_REQUIRED = ["--content", "c.jpg", "--style", "s.jpg", "--out", "o.png"]


def test_unset_tunables_default_to_none():
    args = build_parser().parse_args(_REQUIRED)
    for field in _TUNABLES:
        assert getattr(args, field) is None


def test_tunable_flags_parse():
    args = build_parser().parse_args(
        _REQUIRED + ["--backend", "vgg", "--steps", "3", "--device", "cpu"]
    )
    assert args.backend == "vgg"
    assert args.steps == 3
    assert args.device == "cpu"


def test_unknown_backend_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(_REQUIRED + ["--backend", "bogus"])


def test_compare_rejects_unknown_backend(tmp_path):
    with pytest.raises(SystemExit):
        compare.main([
            "--content", "c.jpg", "--style", "s.jpg",
            "--outdir", str(tmp_path), "--backends", "vgg,bogus",
        ])
