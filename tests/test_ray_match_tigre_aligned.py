"""Tests for TIGRE-aligned NAF ray generator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.ray_utils import compute_near_far_naf, generate_rays_for_view_naf_tigre, sample_points_along_rays
from anatcoder.utils.geometry import CBCTGeometry

try:
    import tigre

    HAS_TIGRE = True
except ImportError:
    HAS_TIGRE = False


def _render_point_projection(geo: CBCTGeometry, point_idx: tuple[int, int, int], angle: float) -> np.ndarray:
    vol = np.zeros(tuple(geo.n_voxel), dtype=np.float32)
    vol[point_idx] = 1.0
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    origins, directions = generate_rays_for_view_naf_tigre(geo, angle=angle, device=torch.device('cpu'))
    near, far = compute_near_far_naf(geo)
    points, step_sizes = sample_points_along_rays(origins, directions, n_samples=256, near=near, far=far, perturb=False)
    vol_size_m = torch.tensor(
        [geo.n_voxel[0] * geo.d_voxel[0] / 1000.0, geo.n_voxel[1] * geo.d_voxel[1] / 1000.0, geo.n_voxel[2] * geo.d_voxel[2] / 1000.0],
        dtype=torch.float32,
    )
    points_norm = points / vol_size_m + 0.5
    grid = (points_norm * 2.0 - 1.0)[..., [2, 1, 0]].reshape(1, 1, -1, 1, 3)
    sampled = F.grid_sample(vol_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    rows, cols = map(int, geo.n_detector)
    proj = (sampled.reshape(rows * cols, -1) * step_sizes).sum(dim=1).reshape(rows, cols).numpy()
    return proj


@pytest.mark.skipif(not HAS_TIGRE, reason='TIGRE not installed')
def test_generate_rays_for_view_naf_tigre_matches_point_peak_angle0() -> None:
    geo = CBCTGeometry()
    tgeo = geo.to_tigre_geometry()
    idx = (geo.n_voxel[0] // 2 + 16, geo.n_voxel[1] // 2, geo.n_voxel[2] // 2)
    vol = np.zeros(tuple(geo.n_voxel), dtype=np.float32)
    vol[idx] = 1.0

    tigre_proj = tigre.Ax(vol, tgeo, np.array([0.0], dtype=np.float32))[0]
    ours_proj = _render_point_projection(geo, idx, angle=0.0)
    peak_t = np.unravel_index(int(np.argmax(tigre_proj)), tigre_proj.shape)
    peak_o = np.unravel_index(int(np.argmax(ours_proj)), ours_proj.shape)
    assert abs(int(peak_t[0]) - int(peak_o[0])) <= 1
    assert abs(int(peak_t[1]) - int(peak_o[1])) <= 1


@pytest.mark.skipif(not HAS_TIGRE, reason='TIGRE not installed')
def test_generate_rays_for_view_naf_tigre_matches_point_peak_quarter_pi() -> None:
    geo = CBCTGeometry()
    tgeo = geo.to_tigre_geometry()
    idx = (geo.n_voxel[0] // 2, geo.n_voxel[1] // 2 + 16, geo.n_voxel[2] // 2)
    angle = float(np.pi / 4.0)
    vol = np.zeros(tuple(geo.n_voxel), dtype=np.float32)
    vol[idx] = 1.0

    tigre_proj = tigre.Ax(vol, tgeo, np.array([angle], dtype=np.float32))[0]
    ours_proj = _render_point_projection(geo, idx, angle=angle)
    peak_t = np.unravel_index(int(np.argmax(tigre_proj)), tigre_proj.shape)
    peak_o = np.unravel_index(int(np.argmax(ours_proj)), ours_proj.shape)
    assert abs(int(peak_t[0]) - int(peak_o[0])) <= 1
    assert abs(int(peak_t[1]) - int(peak_o[1])) <= 1
