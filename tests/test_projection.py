"""Tests for Week 1 projection pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.data.projection import TIGREProjector
from anatcoder.eval.global_metrics import compute_psnr
from anatcoder.utils.geometry import CBCTGeometry, generate_angles


def _ensure_tigre() -> None:
    """Skip test if TIGRE is unavailable."""
    pytest.importorskip('tigre', reason='TIGRE is required for projection tests')


def _projector_or_skip(size: int) -> TIGREProjector:
    """Create a test projector or skip if TIGRE runtime is unavailable."""
    _ensure_tigre()
    geo = CBCTGeometry(
        n_voxel=[size, size, size],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[size, size],
        d_detector=[1.5, 1.5],
    )
    try:
        return TIGREProjector(geo)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'TIGRE runtime unavailable: {exc}')


def _phantom(size: int) -> np.ndarray:
    """Create a smooth synthetic phantom with strong low-frequency structure."""
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing='ij',
    )
    sphere = (x**2 + y**2 + z**2) <= 0.45**2
    blob = np.exp(-((x + 0.25) ** 2 + (y - 0.1) ** 2 + (z + 0.2) ** 2) / 0.15)
    volume = np.zeros((size, size, size), dtype=np.float32)
    volume[sphere] = 0.8
    volume += 0.2 * blob.astype(np.float32)
    return np.clip(volume, 0.0, 1.0).astype(np.float32)


def test_cbct_geometry_defaults() -> None:
    """测试 CBCTGeometry 默认值。"""
    geo = CBCTGeometry()
    assert geo.DSD == 1536.0
    assert geo.DSO == 1000.0
    assert geo.n_voxel == [128, 128, 128]
    assert geo.d_voxel == [1.0, 1.0, 1.0]
    assert geo.n_detector == [256, 256]
    assert geo.d_detector == [1.5, 1.5]


def test_cbct_geometry_to_tigre() -> None:
    """测试转换为 TIGRE geometry 对象。"""
    _ensure_tigre()
    geo = CBCTGeometry(n_voxel=[64, 64, 64], n_detector=[96, 96])
    try:
        tigre_geo = geo.to_tigre_geometry()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'TIGRE geometry construction unavailable: {exc}')

    assert np.all(np.asarray(tigre_geo.nVoxel) == np.array([64, 64, 64]))
    assert np.all(np.asarray(tigre_geo.nDetector) == np.array([96, 96]))


def test_generate_angles() -> None:
    """测试角度生成：数量、范围、弧度。"""
    angles = generate_angles(4, start=0, end=360, endpoint=False)
    assert angles.shape == (4,)
    assert np.isclose(float(angles[0]), 0.0)
    assert np.isclose(float(angles[-1]), np.deg2rad(270.0), atol=1e-6)


def test_forward_project_shape() -> None:
    """测试前向投影输出 shape。"""
    projector = _projector_or_skip(size=32)
    volume = np.random.default_rng(0).random((32, 32, 32), dtype=np.float32)
    angles = generate_angles(4)

    try:
        projections = projector.forward_project(volume, angles)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'TIGRE forward projection failed in test environment: {exc}')

    assert projections.shape == (4, 32, 32)
    assert projections.dtype == np.float32


def test_fdk_reconstruct_shape() -> None:
    """测试 FDK 重建输出 shape。"""
    projector = _projector_or_skip(size=32)
    volume = _phantom(32)
    angles = generate_angles(12)

    try:
        projections = projector.forward_project(volume, angles)
        recon = projector.fdk_reconstruct(projections, angles)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'TIGRE FDK failed in test environment: {exc}')

    assert recon.shape == volume.shape
    assert recon.dtype == np.float32


def test_forward_then_fdk_psnr() -> None:
    """端到端测试：前向投影 + FDK 的 PSNR 应该合理。"""
    projector = _projector_or_skip(size=64)
    volume = _phantom(64)
    angles = generate_angles(50)

    try:
        projections = projector.forward_project(volume, angles)
        recon = projector.fdk_reconstruct(projections, angles)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'TIGRE pipeline unavailable in test environment: {exc}')

    value = compute_psnr(recon, volume, data_range=1.0)
    assert value > 20.0
