"""Tests for anatomy-conditioned VanillaINR behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.network import VanillaINR  # noqa: E402


def test_vanilla_inr_supports_anatomy_condition() -> None:
    """Model should accept anatomy one-hot and return valid attenuation predictions."""
    model = VanillaINR(
        encoder_type='positional',
        n_hidden_layers=2,
        hidden_dim=32,
        last_activation='sigmoid',
        n_anatomy_classes=4,
    )
    coords = torch.rand((17, 3), dtype=torch.float32)
    labels = torch.randint(0, 4, (17,), dtype=torch.int64)
    anatomy = torch.nn.functional.one_hot(labels, num_classes=4).to(dtype=torch.float32)

    out = model(coords, anatomy_onehot=anatomy)
    assert out.shape == (17, 1)
    assert torch.isfinite(out).all()


def test_vanilla_inr_backward_compatible_without_anatomy() -> None:
    """n_anatomy_classes=0 should preserve original forward behavior."""
    model = VanillaINR(
        encoder_type='positional',
        n_hidden_layers=2,
        hidden_dim=32,
        last_activation='sigmoid',
        n_anatomy_classes=0,
    )
    coords = torch.rand((11, 3), dtype=torch.float32)

    out_legacy = model(coords)
    out_explicit_none = model(coords, anatomy_onehot=None)
    assert torch.allclose(out_legacy, out_explicit_none, atol=1e-7)


def test_vanilla_inr_rejects_wrong_anatomy_dim() -> None:
    """Model should validate anatomy class count strictly."""
    model = VanillaINR(
        encoder_type='positional',
        n_hidden_layers=2,
        hidden_dim=32,
        last_activation='sigmoid',
        n_anatomy_classes=3,
    )
    coords = torch.rand((8, 3), dtype=torch.float32)
    bad_anatomy = torch.zeros((8, 4), dtype=torch.float32)
    with pytest.raises(ValueError):
        _ = model(coords, anatomy_onehot=bad_anatomy)


def test_vanilla_inr_accepts_anatomy_labels() -> None:
    """Renderer-facing integer labels should match one-hot conditioning behavior."""
    model = VanillaINR(
        encoder_type='positional',
        n_hidden_layers=2,
        hidden_dim=32,
        last_activation='sigmoid',
        n_anatomy_classes=3,
    )
    coords = torch.rand((10, 3), dtype=torch.float32)
    labels = torch.randint(0, 3, (10,), dtype=torch.long)
    onehot = torch.nn.functional.one_hot(labels, num_classes=3).to(dtype=torch.float32)

    out_labels = model(coords, anatomy_labels=labels)
    out_onehot = model(coords, anatomy_onehot=onehot)
    assert torch.allclose(out_labels, out_onehot, atol=1e-6)
