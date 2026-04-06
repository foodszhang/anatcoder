"""Validate NAF-style ray generation against a reference implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.ray_utils import generate_rays_for_view_naf
from anatcoder.utils.geometry import CBCTGeometry


def _ref_angle2pose(dso_m: float, angle: float) -> np.ndarray:
    phi1 = -np.pi / 2
    R1 = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(phi1), -np.sin(phi1)], [0.0, np.sin(phi1), np.cos(phi1)]],
        dtype=np.float32,
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [[np.cos(phi2), -np.sin(phi2), 0.0], [np.sin(phi2), np.cos(phi2), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    R3 = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot = R3 @ R2 @ R1
    trans = np.array([dso_m * np.cos(angle), dso_m * np.sin(angle), 0.0], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:-1, :-1] = rot
    T[:-1, -1] = trans
    return T


def _reference_rays(geo: CBCTGeometry, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    dsd_m = float(geo.DSD) / 1000.0
    dso_m = float(geo.DSO) / 1000.0
    d_det_u = float(geo.d_detector[0]) / 1000.0
    d_det_v = float(geo.d_detector[1]) / 1000.0
    width = int(geo.n_detector[1])
    height = int(geo.n_detector[0])

    pose = torch.tensor(_ref_angle2pose(dso_m, float(angle)), dtype=torch.float32)
    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width, dtype=torch.float32),
        torch.linspace(0, height - 1, height, dtype=torch.float32),
        indexing='ij',
    )
    uu = (i.t() + 0.5 - width / 2.0) * d_det_u
    vv = (j.t() + 0.5 - height / 2.0) * d_det_v
    dirs = torch.stack((uu / dsd_m, vv / dsd_m, torch.ones_like(uu)), dim=-1)
    rays_d = torch.matmul(pose[:3, :3], dirs.unsqueeze(-1)).squeeze(-1)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    rays_o = pose[:3, -1].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def test_generate_rays_for_view_naf_matches_reference_zero_angle() -> None:
    geo = CBCTGeometry()
    ours_o, ours_d = generate_rays_for_view_naf(geo, angle=0.0, device=torch.device('cpu'))
    ref_o, ref_d = _reference_rays(geo, angle=0.0)
    assert torch.max(torch.abs(ours_o - ref_o)).item() < 1e-5
    assert torch.max(torch.abs(ours_d - ref_d)).item() < 1e-5


def test_generate_rays_for_view_naf_matches_reference_quarter_pi() -> None:
    geo = CBCTGeometry()
    angle = float(np.pi / 4.0)
    ours_o, ours_d = generate_rays_for_view_naf(geo, angle=angle, device=torch.device('cpu'))
    ref_o, ref_d = _reference_rays(geo, angle=angle)
    assert torch.max(torch.abs(ours_o - ref_o)).item() < 1e-5
    assert torch.max(torch.abs(ours_d - ref_d)).item() < 1e-5
