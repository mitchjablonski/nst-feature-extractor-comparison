"""CPU-only unit tests for the Gatys losses (no models, no GPU)."""

import torch

from diffusion_style import losses


def test_gram_matrix_shape_and_symmetry():
    feature = torch.randn(2, 4, 8, 8)
    gram = losses.gram_matrix(feature)
    assert gram.shape == (2, 4, 4)
    assert torch.allclose(gram, gram.transpose(1, 2))


def test_gram_matrix_normalization():
    # All-ones feature: every Gram entry is sum_{hw} 1*1 = H*W, then / (C*H*W) = 1/C.
    feature = torch.ones(1, 3, 5, 5)
    gram = losses.gram_matrix(feature)
    assert torch.allclose(gram, torch.full((1, 3, 3), 1.0 / 3.0))


def test_content_and_style_loss_zero_at_target():
    feats = {"a": torch.randn(1, 4, 6, 6)}
    content_targets = {"a": feats["a"].clone()}
    assert losses.content_loss(feats, content_targets, ("a",)).item() == 0.0
    target_grams = {"a": losses.gram_matrix(feats["a"])}
    assert losses.style_loss(feats, target_grams, ("a",)).item() == 0.0


def test_total_variation_zero_for_constant_image():
    assert losses.total_variation(torch.zeros(1, 3, 4, 4)).item() == 0.0


def test_total_variation_known_value():
    # First column 1, rest 0: only the column-0 -> column-1 horizontal diffs are
    # nonzero (1), i.e. 12 of the 36 dw entries -> dw mean 1/3; dh is 0.
    image = torch.zeros(1, 3, 4, 4)
    image[:, :, :, 0] = 1.0
    assert torch.allclose(losses.total_variation(image), torch.tensor(1.0 / 3.0))
