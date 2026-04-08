"""End-to-end NAF-TIGRE renderer consistency against TIGRE Ax()."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anatcoder.models.ray_utils import compute_near_far_naf, generate_rays_for_view_naf_tigre
from anatcoder.models.renderer import render_rays
from anatcoder.utils.geometry import CBCTGeometry

try:
    import tigre

    HAS_TIGRE = True
except ImportError:
    HAS_TIGRE = False


class GTVolumeOracle(nn.Module):
    """Query densities by trilinear sampling from a fixed GT volume tensor."""

    def __init__(self, volume: np.ndarray, volume_size_world: list[float]):
        super().__init__()
        self.register_buffer('volume', torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0))
        self.bound = None
        self.volume_size_world = volume_size_world
        self.zero_outside_volume = True

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


def _psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(float(data_range)) - 10.0 * np.log10(mse))


@pytest.mark.skipif(not HAS_TIGRE, reason='TIGRE not installed')
def test_render_rays_naf_tigre_matches_tigre_projection() -> None:
    """GT-oracle + render_rays should reproduce TIGRE projections in NAF-TIGRE ray mode."""
    n = 32
    geo = CBCTGeometry(
        n_voxel=[n, n, n],
        d_voxel=[1.0, 1.0, 1.0],
        n_detector=[32, 32],
        d_detector=[1.5, 1.5],
    )
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, n, dtype=np.float32),
        np.linspace(-1.0, 1.0, n, dtype=np.float32),
        np.linspace(-1.0, 1.0, n, dtype=np.float32),
        indexing='ij',
    )
    volume = np.zeros((n, n, n), dtype=np.float32)
    volume[((x + 0.2) ** 2 + (y - 0.1) ** 2 + (z * 0.8) ** 2) < 0.35] = 0.03
    volume[((x - 0.4) ** 2 + (y + 0.35) ** 2 + (z + 0.15) ** 2) < 0.10] = 0.08

    model = GTVolumeOracle(
        volume=volume,
        volume_size_world=[
            float(geo.n_voxel[0] * geo.d_voxel[0]) / 1000.0,
            float(geo.n_voxel[1] * geo.d_voxel[1]) / 1000.0,
            float(geo.n_voxel[2] * geo.d_voxel[2]) / 1000.0,
        ],
    )
    near, far = compute_near_far_naf(geo)
    angles = np.array([0.0, float(np.pi / 4.0)], dtype=np.float32)
    tigre_proj_mm = tigre.Ax(volume, geo.to_tigre_geometry(), angles).astype(np.float32)

    psnr_vals: list[float] = []
    for view_idx, angle in enumerate(angles):
        origins, directions = generate_rays_for_view_naf_tigre(geo, float(angle), device=torch.device('cpu'))
        pred = render_rays(
            model=model,
            ray_origins=origins,
            ray_directions=directions,
            n_samples=192,
            near=near,
            far=far,
            perturb=False,
            chunk_size=8192,
        ).reshape(geo.n_detector[0], geo.n_detector[1])
        gt = tigre_proj_mm[view_idx] / 1000.0
        psnr_vals.append(_psnr(pred.numpy(), gt, data_range=1.0))

    assert min(psnr_vals) > 45.0
