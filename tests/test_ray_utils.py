"""Tests for Week 2 ray utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.ray_utils import (
    generate_rays_for_view,
    normalize_coords,
    sample_points_along_rays,
)
from anatcoder.utils.geometry import CBCTGeometry


def test_generate_rays_for_view_shape() -> None:
    """测试射线生成的输出 shape。"""
    geo = CBCTGeometry()
    origins, directions = generate_rays_for_view(geo, angle=0.0)
    assert origins.shape == (256 * 256, 3)
    assert directions.shape == (256 * 256, 3)


def test_generate_rays_different_angles() -> None:
    """测试不同角度生成的射线不同。"""
    geo = CBCTGeometry()
    origins_0, _ = generate_rays_for_view(geo, angle=0.0)
    origins_90, _ = generate_rays_for_view(geo, angle=float(np.pi / 2))
    assert not torch.allclose(origins_0[0], origins_90[0])


def test_ray_origin_distance() -> None:
    """测试射线原点到旋转中心的距离 = DSO。"""
    geo = CBCTGeometry()
    origins, _ = generate_rays_for_view(geo, angle=0.0)
    dists = torch.linalg.norm(origins, dim=-1)
    assert torch.allclose(dists, torch.full_like(dists, float(geo.DSO)), atol=1e-4)


def test_ray_direction_normalized() -> None:
    """测试 ray_directions 已归一化（模长 ≈ 1）。"""
    geo = CBCTGeometry()
    _, directions = generate_rays_for_view(geo, angle=0.0)
    norms = torch.linalg.norm(directions, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sample_points_shape() -> None:
    """测试采样点 shape = [N, n_samples, 3]。"""
    origins = torch.zeros((8, 3), dtype=torch.float32)
    dirs = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32).repeat(8, 1)
    points, step_sizes = sample_points_along_rays(origins, dirs, n_samples=16, near=1.0, far=9.0, perturb=False)
    assert points.shape == (8, 16, 3)
    assert step_sizes.shape == (8, 16)


def test_sample_points_range() -> None:
    """测试采样点在 [near, far] 范围内。"""
    origins = torch.zeros((4, 3), dtype=torch.float32)
    dirs = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(4, 1)
    points, _ = sample_points_along_rays(origins, dirs, n_samples=32, near=2.0, far=6.0, perturb=False)
    t = points[..., 2]
    assert torch.all(t >= 2.0)
    assert torch.all(t <= 6.0)


def test_normalize_coords() -> None:
    """测试坐标归一化：volume 中心 -> (0.5, 0.5, 0.5)。"""
    center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    norm = normalize_coords(center, [128.0, 128.0, 128.0])
    assert torch.allclose(norm, torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32))


def test_normalize_coords_boundary() -> None:
    """测试 volume 边界坐标归一化为 0 和 1。"""
    points = torch.tensor(
        [
            [-64.0, -32.0, -16.0],
            [64.0, 32.0, 16.0],
        ],
        dtype=torch.float32,
    )
    norm = normalize_coords(points, [128.0, 64.0, 32.0])
    expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(norm, expected, atol=1e-6)
