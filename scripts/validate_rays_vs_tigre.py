"""Validate ray generators against TIGRE with point-source and projection tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from anatcoder.models.ray_utils import (
    compute_near_far_naf,
    generate_rays_for_view,
    generate_rays_for_view_naf,
    generate_rays_for_view_naf_tigre,
    normalize_coords,
    sample_points_along_rays,
)
from anatcoder.utils.geometry import CBCTGeometry


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    use_meter: bool
    fn: callable


def _render_projection(
    volume: np.ndarray,
    geo: CBCTGeometry,
    angle: float,
    generator: GeneratorSpec,
    n_samples: int,
) -> np.ndarray:
    """Render one projection from a GT volume using a specific ray generator."""
    vol_t = torch.from_numpy(volume.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    origins, directions = generator.fn(geo, float(angle), device=torch.device('cpu'))

    if generator.use_meter:
        near, far = compute_near_far_naf(geo)
        vol_size = torch.tensor(
            [
                geo.n_voxel[0] * geo.d_voxel[0] / 1000.0,
                geo.n_voxel[1] * geo.d_voxel[1] / 1000.0,
                geo.n_voxel[2] * geo.d_voxel[2] / 1000.0,
            ],
            dtype=torch.float32,
        )
        points, step_sizes = sample_points_along_rays(origins, directions, n_samples, near, far, perturb=False)
        points_norm = points / vol_size + 0.5
    else:
        vol_size_mm = [
            float(geo.n_voxel[0] * geo.d_voxel[0]),
            float(geo.n_voxel[1] * geo.d_voxel[1]),
            float(geo.n_voxel[2] * geo.d_voxel[2]),
        ]
        diag = float(np.linalg.norm(np.asarray(vol_size_mm, dtype=np.float32)))
        near = float(geo.DSO - 0.5 * diag)
        far = float(geo.DSO + 0.5 * diag)
        points, step_sizes = sample_points_along_rays(origins, directions, n_samples, near, far, perturb=False)
        points_norm = normalize_coords(points, vol_size_mm)

    grid = (points_norm * 2.0 - 1.0)[..., [2, 1, 0]].reshape(1, 1, -1, 1, 3)
    sampled = F.grid_sample(
        vol_t,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    rows, cols = int(geo.n_detector[0]), int(geo.n_detector[1])
    proj = (sampled.reshape(rows * cols, -1) * step_sizes).sum(dim=1).reshape(rows, cols)
    return proj.cpu().numpy().astype(np.float32, copy=False)


def _psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(float(data_range)) - 10.0 * np.log10(mse))


def _point_source_checks(geo: CBCTGeometry, generators: list[GeneratorSpec]) -> None:
    import tigre

    tgeo = geo.to_tigre_geometry()
    angle = np.array([0.0], dtype=np.float32)
    n1, n2, n3 = map(int, geo.n_voxel)
    c1, c2, c3 = n1 // 2, n2 // 2, n3 // 2
    probes = {
        'center': (c1, c2, c3),
        'axis0+': (min(c1 + n1 // 4, n1 - 1), c2, c3),
        'axis1+': (c1, min(c2 + n2 // 4, n2 - 1), c3),
        'axis2+': (c1, c2, min(c3 + n3 // 4, n3 - 1)),
    }

    print('=== 1) Point-source peak checks (angle=0) ===')
    for tag, idx in probes.items():
        vol = np.zeros(tuple(map(int, geo.n_voxel)), dtype=np.float32)
        vol[idx] = 1.0
        tigre_proj = tigre.Ax(vol, tgeo, angle)[0]
        tigre_peak = np.unravel_index(int(np.argmax(tigre_proj)), tigre_proj.shape)
        print(f'[{tag}] TIGRE peak row={tigre_peak[0]}, col={tigre_peak[1]}')
        for gen in generators:
            ours = _render_projection(vol, geo, 0.0, gen, n_samples=384)
            ours_peak = np.unravel_index(int(np.argmax(ours)), ours.shape)
            dr = abs(int(ours_peak[0]) - int(tigre_peak[0]))
            dc = abs(int(ours_peak[1]) - int(tigre_peak[1]))
            print(f'  - {gen.name:16s} row={ours_peak[0]}, col={ours_peak[1]} (|dr|={dr}, |dc|={dc})')


def _projection_psnr_checks(
    geo: CBCTGeometry,
    case: str,
    n_views: int,
    n_samples: int,
    generators: list[GeneratorSpec],
) -> None:
    angles = np.load(f'data/projections/{case}/{n_views}views/angles.npy').astype(np.float32)
    gt_projs_mm = np.load(f'data/projections/{case}/{n_views}views/projections.npy').astype(np.float32)
    volume = np.load(f'data/processed/{case}/volume.npy').astype(np.float32)

    print('\n=== 2) GT-volume rendered projections vs TIGRE projections ===')
    for gen in generators:
        psnr_values: list[float] = []
        for view_idx in range(len(angles)):
            pred = _render_projection(volume, geo, float(angles[view_idx]), gen, n_samples=n_samples)
            if gen.use_meter:
                gt = gt_projs_mm[view_idx] / 1000.0
            else:
                gt = gt_projs_mm[view_idx]
            psnr_values.append(_psnr(pred, gt, data_range=1.0))
        arr = np.asarray(psnr_values, dtype=np.float32)
        print(
            f'- {gen.name:16s}: mean={float(arr.mean()):.4f} dB, '
            f'min={float(arr.min()):.4f}, max={float(arr.max()):.4f}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate ray generators against TIGRE.')
    parser.add_argument('--case', type=str, default='case001')
    parser.add_argument('--n-views', type=int, default=50)
    parser.add_argument('--n-samples', type=int, default=384)
    args = parser.parse_args()

    geo = CBCTGeometry()
    generators = [
        GeneratorSpec(name='legacy_mm', use_meter=False, fn=generate_rays_for_view),
        GeneratorSpec(name='naf_raw_m', use_meter=True, fn=generate_rays_for_view_naf),
        GeneratorSpec(name='naf_tigre_m', use_meter=True, fn=generate_rays_for_view_naf_tigre),
    ]
    _point_source_checks(geo, generators)
    _projection_psnr_checks(geo, args.case, args.n_views, args.n_samples, generators)


if __name__ == '__main__':
    main()
