"""Tests for NAF near/far distance computation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.ray_utils import compute_near_far_naf
from anatcoder.utils.geometry import CBCTGeometry


def _reference_near_far(geo: CBCTGeometry, tolerance: float = 0.005) -> tuple[float, float]:
    dso = float(geo.DSO) / 1000.0
    s_voxel = np.asarray(geo.n_voxel, dtype=np.float32) * (np.asarray(geo.d_voxel, dtype=np.float32) / 1000.0)
    half_x = float(s_voxel[0]) * 0.5
    half_y = float(s_voxel[1]) * 0.5
    dist1 = np.linalg.norm([-half_x, -half_y])
    dist2 = np.linalg.norm([-half_x, +half_y])
    dist3 = np.linalg.norm([+half_x, -half_y])
    dist4 = np.linalg.norm([+half_x, +half_y])
    dist_max = float(np.max([dist1, dist2, dist3, dist4]))
    near = max(0.0, dso - dist_max - tolerance)
    far = min(dso * 2.0, dso + dist_max + tolerance)
    return near, far


def test_compute_near_far_naf_matches_reference() -> None:
    geo = CBCTGeometry(
        DSD=1536.0,
        DSO=1000.0,
        n_voxel=[128, 128, 128],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[256, 256],
        d_detector=[1.5, 1.5],
    )
    near, far = compute_near_far_naf(geo)
    ref_near, ref_far = _reference_near_far(geo)
    assert abs(near - ref_near) < 1e-8
    assert abs(far - ref_far) < 1e-8
    assert near >= 0.0
    assert far > near
