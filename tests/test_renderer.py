"""Tests for Week 2 volume renderer and reconstruction utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.renderer import VolumeRenderer, reconstruct_volume, render_rays


class ConstantModel(nn.Module):
    """Dummy model returning constant attenuation."""

    def __init__(self, value: float, volume_size_mm: list[float]):
        super().__init__()
        self.value = float(value)
        self.volume_size_mm = volume_size_mm

    def query_density(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.full((coords.shape[0], 1), self.value, dtype=coords.dtype, device=coords.device)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.query_density(coords)


def test_volume_renderer_shape() -> None:
    """测试体渲染器输出 shape。"""
    renderer = VolumeRenderer()
    densities = torch.rand((100, 64, 1), dtype=torch.float32)
    step_sizes = torch.rand((100, 64), dtype=torch.float32)
    out = renderer(densities, step_sizes)
    assert out.shape == (100,)


def test_volume_renderer_zero_density() -> None:
    """测试零密度 -> 零 line integral。"""
    renderer = VolumeRenderer()
    densities = torch.zeros((32, 100, 1), dtype=torch.float32)
    step_sizes = torch.full((32, 100), 0.1, dtype=torch.float32)
    out = renderer(densities, step_sizes)
    assert torch.allclose(out, torch.zeros_like(out))


def test_volume_renderer_uniform_density() -> None:
    """测试均匀密度积分值。"""
    renderer = VolumeRenderer()
    densities = torch.ones((16, 100, 1), dtype=torch.float32)
    step_sizes = torch.full((16, 100), 0.1, dtype=torch.float32)
    out = renderer(densities, step_sizes)
    assert torch.allclose(out, torch.full((16,), 10.0), atol=1e-6)


def test_volume_renderer_gradient() -> None:
    """测试梯度可以回传。"""
    renderer = VolumeRenderer()
    densities = torch.rand((8, 32, 1), dtype=torch.float32, requires_grad=True)
    step_sizes = torch.rand((8, 32), dtype=torch.float32)
    out = renderer(densities, step_sizes).sum()
    out.backward()
    assert densities.grad is not None
    assert torch.isfinite(densities.grad).all()


def test_render_rays_with_constant_model() -> None:
    """常数密度模型渲染结果应接近 mu*(far-near)。"""
    model = ConstantModel(value=0.01, volume_size_mm=[128.0, 128.0, 128.0])
    origins = torch.zeros((64, 3), dtype=torch.float32)
    directions = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32).repeat(64, 1)
    out = render_rays(
        model,
        origins,
        directions,
        n_samples=128,
        near=0.0,
        far=100.0,
        perturb=False,
        chunk_size=256,
    )
    assert out.shape == (64,)
    assert torch.allclose(out, torch.full((64,), 1.0, dtype=torch.float32), atol=1e-5)


def test_reconstruct_volume_shape() -> None:
    """测试 volume 重建输出 shape。"""
    model = ConstantModel(value=0.02, volume_size_mm=[16.0, 16.0, 16.0])
    volume = reconstruct_volume(
        model=model,
        volume_size=[16, 16, 16],
        voxel_size=[1.0, 1.0, 1.0],
        chunk_size=2048,
        device=torch.device('cpu'),
    )
    assert volume.shape == (16, 16, 16)
    assert volume.dtype == np.float32
    assert np.allclose(volume, 0.02, atol=1e-6)
