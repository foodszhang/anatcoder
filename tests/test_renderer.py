"""Tests for Week 2 volume renderer and reconstruction utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CaptureInputModel(nn.Module):
    """Model that records queried coords for normalization-path assertions."""

    def __init__(
        self,
        *,
        volume_size_world: list[float] | None = None,
        volume_size_mm: list[float] | None = None,
        coord_axis_mode: str = 'identity',
        zero_outside_volume: bool = False,
    ):
        super().__init__()
        if volume_size_world is not None:
            self.volume_size_world = volume_size_world
        if volume_size_mm is not None:
            self.volume_size_mm = volume_size_mm
        self.coord_axis_mode = coord_axis_mode
        self.zero_outside_volume = bool(zero_outside_volume)
        self.captured: torch.Tensor | None = None

    def query_density(self, coords: torch.Tensor) -> torch.Tensor:
        self.captured = coords.detach().clone()
        return torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.query_density(coords)


class OnesModel(nn.Module):
    """Model returning one-valued density everywhere."""

    def __init__(self, volume_size_world: list[float], zero_outside_volume: bool):
        super().__init__()
        self.volume_size_world = volume_size_world
        self.zero_outside_volume = bool(zero_outside_volume)

    def query_density(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.query_density(coords)


class GridOracleModel(nn.Module):
    """Oracle model using trilinear sampling from a fixed volume."""

    def __init__(self, volume: np.ndarray, volume_size_world: list[float], coord_axis_mode: str = 'identity'):
        super().__init__()
        self.register_buffer('volume', torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0))
        self.volume_size_world = volume_size_world
        self.coord_axis_mode = coord_axis_mode

    def query_density(self, coords: torch.Tensor) -> torch.Tensor:
        grid = (coords * 2.0 - 1.0)[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.volume.to(coords.device),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return sampled.reshape(-1, 1)

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


def test_render_rays_applies_line_integral_scale() -> None:
    """Renderer should apply model-configured line-integral step scale."""
    model = ConstantModel(value=0.01, volume_size_mm=[128.0, 128.0, 128.0])
    model.line_integral_scale = 1000.0
    origins = torch.zeros((8, 3), dtype=torch.float32)
    directions = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32).repeat(8, 1)
    out = render_rays(
        model,
        origins,
        directions,
        n_samples=32,
        near=0.0,
        far=0.1,
        perturb=False,
        chunk_size=128,
    )
    assert out.shape == (8,)
    assert torch.allclose(out, torch.full((8,), 1.0, dtype=torch.float32), atol=1e-5)


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


def test_render_rays_naf_maps_to_unit_interval() -> None:
    """World-size normalization should map NAF points into [0,1]."""
    model = CaptureInputModel(
        volume_size_world=[0.128, 0.128, 0.128],
        zero_outside_volume=True,
    )
    origins = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    _ = render_rays(
        model,
        origins,
        directions,
        n_samples=8,
        near=-0.064,
        far=0.064,
        perturb=False,
        chunk_size=64,
    )
    assert model.captured is not None
    assert torch.all(model.captured >= 0.0)
    assert torch.all(model.captured <= 1.0)


def test_reconstruct_volume_naf_maps_to_unit_interval() -> None:
    """NAF reconstruction should use same world-size normalization into [0,1]."""
    model = CaptureInputModel(
        volume_size_world=[0.004, 0.004, 0.004],
        coord_axis_mode='identity',
        zero_outside_volume=True,
    )
    _ = reconstruct_volume(
        model=model,
        volume_size=[4, 4, 4],
        voxel_size=[0.001, 0.001, 0.001],
        chunk_size=1024,
        device=torch.device('cpu'),
    )
    assert model.captured is not None
    assert torch.all(model.captured >= 0.0)
    assert torch.all(model.captured <= 1.0)


def test_render_rays_zero_outside_volume_masks_samples() -> None:
    """When zero_outside_volume is enabled, out-of-bounds samples should not contribute."""
    model_masked = OnesModel(volume_size_world=[1.0, 1.0, 1.0], zero_outside_volume=True)
    model_unmasked = OnesModel(volume_size_world=[1.0, 1.0, 1.0], zero_outside_volume=False)
    origins = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    masked = render_rays(
        model_masked,
        origins,
        directions,
        n_samples=16,
        near=0.0,
        far=1.0,
        perturb=False,
        chunk_size=64,
    )
    unmasked = render_rays(
        model_unmasked,
        origins,
        directions,
        n_samples=16,
        near=0.0,
        far=1.0,
        perturb=False,
        chunk_size=64,
    )
    assert torch.allclose(masked, torch.zeros_like(masked), atol=1e-6)
    assert torch.allclose(unmasked, torch.ones_like(unmasked), atol=1e-6)


def test_reconstruct_volume_identity_axis_mode_matches_oracle_grid() -> None:
    """Identity axis-mode should reconstruct a sampled oracle grid consistently."""
    n = 6
    volume = (np.arange(n**3, dtype=np.float32).reshape(n, n, n) / float(n**3)).astype(np.float32)
    model = GridOracleModel(volume=volume, volume_size_world=[float(n - 1), float(n - 1), float(n - 1)])
    recon = reconstruct_volume(
        model=model,
        volume_size=[n, n, n],
        voxel_size=[1.0, 1.0, 1.0],
        chunk_size=1024,
        device=torch.device('cpu'),
    )
    assert np.max(np.abs(recon - volume)) < 1e-5
