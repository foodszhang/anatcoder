"""Unit tests for anatomy-aware TV regularization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.losses.anatomy_tv import anatomy_tv_loss  # noqa: E402


class ConstantModel(nn.Module):
    def __init__(self, value: float = 0.5) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(value), dtype=torch.float32))

    def query_density(self, coords: torch.Tensor, anatomy_labels: torch.Tensor | None = None) -> torch.Tensor:
        _ = coords
        _ = anatomy_labels
        return self.bias.expand(coords.shape[0], 1)


class CoordLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def query_density(self, coords: torch.Tensor, anatomy_labels: torch.Tensor | None = None) -> torch.Tensor:
        _ = anatomy_labels
        return (coords.sum(dim=-1, keepdim=True) * self.scale).to(dtype=coords.dtype)


class LabelOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def query_density(self, coords: torch.Tensor, anatomy_labels: torch.Tensor | None = None) -> torch.Tensor:
        _ = coords
        if anatomy_labels is None:
            raise ValueError('anatomy_labels must be provided')
        return anatomy_labels.to(dtype=torch.float32).unsqueeze(-1) * self.gain


def test_anatomy_tv_loss_zero_for_constant_field() -> None:
    seg = torch.zeros((8, 8, 8), dtype=torch.long)
    model = ConstantModel(value=0.25)

    torch.manual_seed(0)
    loss = anatomy_tv_loss(
        model=model,
        seg_volume=seg,
        volume_size_world=[0.008, 0.008, 0.008],
        n_sample_pairs=2048,
        alpha=0.05,
        n_anatomy_classes=0,
        device='cpu',
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) == 0.0


def test_anatomy_tv_loss_reduces_when_alpha_is_small() -> None:
    d = 16
    seg_all_same = torch.zeros((d, d, d), dtype=torch.long)
    seg_checker = torch.arange(d * d * d, dtype=torch.long).reshape(d, d, d) % 2
    model = CoordLinearModel()

    torch.manual_seed(7)
    loss_same = anatomy_tv_loss(
        model=model,
        seg_volume=seg_all_same,
        volume_size_world=[0.016, 0.016, 0.016],
        n_sample_pairs=4096,
        alpha=0.05,
        n_anatomy_classes=2,
        device='cpu',
    )
    torch.manual_seed(7)
    loss_checker = anatomy_tv_loss(
        model=model,
        seg_volume=seg_checker,
        volume_size_world=[0.016, 0.016, 0.016],
        n_sample_pairs=4096,
        alpha=0.05,
        n_anatomy_classes=2,
        device='cpu',
    )
    assert torch.isfinite(loss_same)
    assert torch.isfinite(loss_checker)
    assert float(loss_checker.item()) < float(loss_same.item())


def test_anatomy_tv_loss_passes_labels_to_model() -> None:
    # Checkerboard ensures adjacent labels differ along any axis.
    d = 10
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(d, dtype=torch.long),
            torch.arange(d, dtype=torch.long),
            torch.arange(d, dtype=torch.long),
            indexing='ij',
        ),
        dim=0,
    )
    seg = (grid.sum(dim=0) % 2).to(torch.long)
    model = LabelOnlyModel()

    torch.manual_seed(123)
    loss = anatomy_tv_loss(
        model=model,
        seg_volume=seg,
        volume_size_world=[0.010, 0.010, 0.010],
        n_sample_pairs=2048,
        alpha=0.0,
        n_anatomy_classes=2,
        device='cpu',
    )
    assert torch.isfinite(loss)
    # All adjacent pairs are cross-class and alpha=0 -> fully suppressed TV.
    assert float(loss.item()) == 0.0
