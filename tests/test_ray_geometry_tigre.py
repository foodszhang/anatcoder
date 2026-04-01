"""Validate ray_utils geometry against TIGRE forward projection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.ray_utils import generate_rays_for_view, normalize_coords, sample_points_along_rays
from anatcoder.models.renderer import VolumeRenderer
from anatcoder.utils.geometry import CBCTGeometry

try:
    import tigre

    HAS_TIGRE = True
except ImportError:
    HAS_TIGRE = False


def _make_sphere_phantom(n: int = 64) -> np.ndarray:
    """Create a centered sphere phantom with constant attenuation."""
    coords = np.linspace(-1, 1, n, dtype=np.float32)
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    r2 = xx**2 + yy**2 + zz**2
    vol = np.zeros((n, n, n), dtype=np.float32)
    vol[r2 < 0.6] = 0.02
    return vol


def _render_with_ours(
    phantom: np.ndarray,
    geo: CBCTGeometry,
    angle: float,
    n_samples: int = 512,
) -> np.ndarray:
    """Render one projection with ray_utils + grid_sample + VolumeRenderer."""
    origins, directions = generate_rays_for_view(geo, angle=angle, device=torch.device("cpu"))
    n = int(phantom.shape[0])
    diag = float(np.linalg.norm(np.array([n, n, n], dtype=np.float32)))
    near = float(geo.DSO - 0.5 * diag)
    far = float(geo.DSO + 0.5 * diag)

    points, step_sizes = sample_points_along_rays(
        origins,
        directions,
        n_samples=n_samples,
        near=near,
        far=far,
        perturb=False,
    )
    points_norm = normalize_coords(points, [float(n), float(n), float(n)])

    phantom_tensor = torch.from_numpy(phantom).unsqueeze(0).unsqueeze(0)
    grid = (points_norm * 2.0 - 1.0)[..., [2, 1, 0]]
    n_rays, n_samples_per_ray = points_norm.shape[0], points_norm.shape[1]
    flat_grid = grid.reshape(1, n_rays * n_samples_per_ray, 1, 1, 3)
    sampled = F.grid_sample(
        phantom_tensor,
        flat_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    densities = sampled.reshape(n_rays, n_samples_per_ray, 1)
    return VolumeRenderer()(densities, step_sizes).detach().numpy()


@pytest.mark.skipif(not HAS_TIGRE, reason="TIGRE not installed")
def test_ray_projection_matches_tigre_single_angle() -> None:
    """Compare single-angle projection distribution against TIGRE."""
    n = 64
    phantom = _make_sphere_phantom(n)
    geo = CBCTGeometry(
        DSD=1536.0,
        DSO=1000.0,
        n_voxel=[n, n, n],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[64, 64],
        d_detector=[1.5, 1.5],
    )
    tigre_proj = tigre.Ax(phantom, geo.to_tigre_geometry(), np.array([0.0], dtype=np.float32))[0].flatten()
    our_proj = _render_with_ours(phantom, geo, angle=0.0)

    our_mean = float(np.mean(our_proj[our_proj > 0]))
    tigre_mean = float(np.mean(tigre_proj[tigre_proj > 0]))
    assert abs(our_mean - tigre_mean) / (tigre_mean + 1e-8) < 0.5

    our_sorted = np.sort(our_proj)
    tigre_sorted = np.sort(tigre_proj)
    rmse = float(np.sqrt(np.mean((our_sorted - tigre_sorted) ** 2)))
    nrmse = rmse / (float(np.max(tigre_proj)) + 1e-8)
    assert nrmse < 0.10


@pytest.mark.skipif(not HAS_TIGRE, reason="TIGRE not installed")
def test_ray_projection_matches_tigre_multi_angle() -> None:
    """Compare projection distributions at multiple cardinal angles."""
    n = 64
    phantom = _make_sphere_phantom(n)
    geo = CBCTGeometry(
        DSD=1536.0,
        DSO=1000.0,
        n_voxel=[n, n, n],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[64, 64],
        d_detector=[1.5, 1.5],
    )
    tigre_geo = geo.to_tigre_geometry()

    for deg in [0, 90, 180, 270]:
        angle = np.deg2rad(np.array([deg], dtype=np.float32))[0]
        tigre_proj = tigre.Ax(phantom, tigre_geo, np.array([angle], dtype=np.float32))[0].flatten()
        our_proj = _render_with_ours(phantom, geo, angle=float(angle))

        our_sorted = np.sort(our_proj)
        tigre_sorted = np.sort(tigre_proj)
        rmse = float(np.sqrt(np.mean((our_sorted - tigre_sorted) ** 2)))
        nrmse = rmse / (float(np.max(tigre_proj)) + 1e-8)
        assert nrmse < 0.10, f"Angle {deg}deg: NRMSE={nrmse:.4f} > 0.10"


@pytest.mark.skipif(not HAS_TIGRE, reason="TIGRE not installed")
def test_detector_pixel_ordering_matches_tigre() -> None:
    """Validate detector pixel ordering by direct 2D projection comparison."""
    n = 64
    phantom = np.zeros((n, n, n), dtype=np.float32)
    phantom[10:20, 10:20, 40:50] = 0.05
    phantom[40:50, 40:50, 10:20] = 0.02

    geo = CBCTGeometry(
        DSD=1536.0,
        DSO=1000.0,
        n_voxel=[n, n, n],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[64, 64],
        d_detector=[1.5, 1.5],
    )
    angle = 0.3
    tigre_proj = tigre.Ax(phantom, geo.to_tigre_geometry(), np.array([angle], dtype=np.float32))[0]
    our_proj = _render_with_ours(phantom, geo, angle=float(angle)).reshape(64, 64)

    corr = float(np.corrcoef(our_proj.flatten(), tigre_proj.flatten())[0, 1])
    assert corr > 0.95, f"Pixel-wise correlation too low: {corr:.4f}"

    rmse = float(np.sqrt(np.mean((our_proj - tigre_proj) ** 2)))
    nrmse = rmse / (float(np.max(tigre_proj)) + 1e-8)
    assert nrmse < 0.10, f"Pixel-wise NRMSE={nrmse:.4f} > 0.10"

