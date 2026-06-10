"""Preset/config invariants: every advertised backend must build a valid Config."""

import pytest

from diffusion_style.backends import BACKENDS
from diffusion_style.config import backend_config, ddpm_config


def test_every_backend_has_a_working_preset():
    for name in BACKENDS:
        config = backend_config(name)
        assert config.backend == name
        # Layer names must be hookable strings, and weights sane.
        assert config.style_layers and config.content_layers
        assert config.style_weight > 0 and config.steps > 0


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend preset"):
        backend_config("bogus")


def test_overrides_take_precedence_over_preset():
    config = backend_config("vgg", steps=7, style_weight=123.0)
    assert config.steps == 7
    assert config.style_weight == 123.0
    assert config.optimizer == "lbfgs"  # untouched preset value survives


def test_ddpm_shim():
    assert ddpm_config().backend == "ddpm"
