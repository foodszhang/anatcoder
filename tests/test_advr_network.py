"""Tests for ADVR shared-backbone + class-specific-head model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.advr_network import ADVRNetwork  # noqa: E402


def _make_constant_advr(values: list[float]) -> ADVRNetwork:
    model = ADVRNetwork(
        encoder_type='positional',
        n_hidden_layers=1,
        hidden_dim=8,
        head_hidden_dim=4,
        n_anatomy_classes=len(values),
        last_activation='sigmoid',
    )
    with torch.no_grad():
        # Shared backbone -> zeros.
        first = model.shared_mlp[0]
        first.weight.zero_()
        first.bias.zero_()
        # Each head emits a class-specific logit constant.
        for class_idx, target in enumerate(values):
            h0 = model.heads[class_idx][0]
            h1 = model.heads[class_idx][2]
            h0.weight.zero_()
            h0.bias.zero_()
            h1.weight.zero_()
            logit = torch.logit(torch.tensor(float(target), dtype=torch.float32), eps=1e-6)
            h1.bias.fill_(float(logit))
    return model


def test_advr_routes_points_to_class_heads() -> None:
    """Each sampled label should route to its class-specific attenuation head."""
    targets = [0.2, 0.5, 0.8]
    model = _make_constant_advr(targets)
    coords = torch.rand((9, 3), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 2, 1, 0, 0, 2, 1], dtype=torch.long)

    out = model.query_density(coords, anatomy_labels=labels).squeeze(-1)
    expected = torch.tensor([targets[int(i)] for i in labels.tolist()], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6)


def test_advr_falls_back_to_head_zero_without_labels() -> None:
    """Without anatomy labels, ADVR should use class-0 head as deterministic fallback."""
    targets = [0.3, 0.7]
    model = _make_constant_advr(targets)
    coords = torch.rand((5, 3), dtype=torch.float32)

    out = model.query_density(coords).squeeze(-1)
    assert torch.allclose(out, torch.full((5,), targets[0], dtype=torch.float32), atol=1e-6)


def test_advr_query_density_accepts_onehot() -> None:
    """One-hot anatomy conditions should map to identical class routing."""
    targets = [0.25, 0.55, 0.9]
    model = _make_constant_advr(targets)
    coords = torch.rand((4, 3), dtype=torch.float32)
    labels = torch.tensor([2, 1, 0, 2], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(labels, num_classes=3).to(dtype=torch.float32)

    out = model.query_density(coords, anatomy_onehot=onehot).squeeze(-1)
    expected = torch.tensor([targets[int(i)] for i in labels.tolist()], dtype=torch.float32)
    assert torch.allclose(out, expected, atol=1e-6)
