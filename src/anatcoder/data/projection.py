"""TIGRE projection and reconstruction wrappers for sparse-view CBCT."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from anatcoder.eval.global_metrics import evaluate_reconstruction
from anatcoder.utils.geometry import CBCTGeometry, generate_angles
from anatcoder.utils.io import load_numpy, save_numpy

console = Console()


class TIGREProjector:
    """TIGRE v3 projector wrapper for forward and inverse CBCT operations."""

    def __init__(self, geo: CBCTGeometry | Mapping[str, Any]) -> None:
        """Initialize TIGRE projector from ``CBCTGeometry`` or geometry dict."""
        if isinstance(geo, CBCTGeometry):
            self.geo = geo
        elif isinstance(geo, Mapping):
            self.geo = CBCTGeometry(
                DSD=float(geo.get('DSD', 1536.0)),
                DSO=float(geo.get('DSO', 1000.0)),
                n_voxel=list(geo.get('n_voxel', [128, 128, 128])),
                d_voxel=list(geo.get('d_voxel', [1.0, 1.0, 1.0])),
                n_detector=list(
                    geo.get('n_detector', geo.get('detector_size', [256, 256]))
                ),
                d_detector=list(
                    geo.get('d_detector', geo.get('detector_spacing', [1.5, 1.5]))
                ),
            )
        else:
            raise TypeError(f'Unsupported geometry type: {type(geo)}')

        try:
            import tigre
        except ImportError as exc:
            raise ImportError(
                'TIGRE is not installed. Install with `uv pip install tigre` '
                'and ensure CUDA runtime is available.'
            ) from exc

        self._tigre = tigre
        self.tigre_geo = self.geo.to_tigre_geometry()

    def forward_project(self, volume: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Forward-project ``[D, H, W]`` volume into detector projections.

        Args:
            volume: Volume in ``[z, y, x]`` order.
            angles: View angles in radians, shape ``[N]``.

        Returns:
            Projection stack with shape ``[N, detector_rows, detector_cols]``.
        """
        if volume.ndim != 3:
            raise ValueError(f'Volume must be 3D, got shape: {volume.shape}')

        volume_f32 = np.ascontiguousarray(volume.astype(np.float32, copy=False))
        angles_f32 = np.ascontiguousarray(np.asarray(angles, dtype=np.float32))

        try:
            projections = self._tigre.Ax(volume_f32, self.tigre_geo, angles_f32)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f'TIGRE forward projection failed: {exc}') from exc

        return np.asarray(projections, dtype=np.float32)

    def fdk_reconstruct(self, projections: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Reconstruct volume with FDK algorithm.

        Args:
            projections: Projection stack ``[N, rows, cols]``.
            angles: Angles in radians, shape ``[N]``.

        Returns:
            Reconstructed volume ``[D, H, W]``.
        """
        proj_f32 = np.ascontiguousarray(projections.astype(np.float32, copy=False))
        angles_f32 = np.ascontiguousarray(np.asarray(angles, dtype=np.float32))

        try:
            recon = self._tigre.algorithms.fdk(proj_f32, self.tigre_geo, angles_f32)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f'TIGRE FDK reconstruction failed: {exc}') from exc

        return np.asarray(recon, dtype=np.float32)

    def sart_reconstruct(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        n_iter: int = 50,
    ) -> np.ndarray:
        """Reconstruct volume with SART algorithm.

        Args:
            projections: Projection stack ``[N, rows, cols]``.
            angles: Angles in radians, shape ``[N]``.
            n_iter: Number of SART iterations.

        Returns:
            Reconstructed volume ``[D, H, W]``.
        """
        proj_f32 = np.ascontiguousarray(projections.astype(np.float32, copy=False))
        angles_f32 = np.ascontiguousarray(np.asarray(angles, dtype=np.float32))

        try:
            recon = self._tigre.algorithms.sart(proj_f32, self.tigre_geo, angles_f32, niter=int(n_iter))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f'TIGRE SART reconstruction failed: {exc}') from exc

        return np.asarray(recon, dtype=np.float32)


def generate_sparse_projections(
    volume_path: str,
    n_views_list: list[int],
    output_dir: str,
    geo: CBCTGeometry,
) -> dict:
    """Generate sparse projections and FDK baselines for one processed CT volume.

    For each ``n_views`` value this function:
      1. Creates uniform scan angles.
      2. Runs forward projection.
      3. Runs FDK reconstruction.
      4. Computes global metrics against GT.
      5. Saves ``projections.npy``, ``angles.npy``, ``fdk_recon.npy`` and ``metrics.json``.

    Returns:
        Summary dictionary keyed by view count.
    """
    volume_file = Path(volume_path)
    if not volume_file.exists():
        raise FileNotFoundError(f'Volume file not found: {volume_file}')

    if not n_views_list:
        raise ValueError('n_views_list must not be empty')

    gt_volume = np.asarray(load_numpy(volume_file), dtype=np.float32)
    case_name = volume_file.parent.name

    out_root = Path(output_dir)
    case_dir = out_root if out_root.name == case_name else out_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    projector = TIGREProjector(geo)
    summary: dict[str, Any] = {'case_name': case_name, 'results': {}}

    for n_views in n_views_list:
        if n_views <= 0:
            raise ValueError(f'n_views must be positive, got: {n_views}')

        console.print(f'[cyan]Generating[/cyan] case={case_name}, views={n_views}')
        angles = generate_angles(n_views=n_views)
        projections = projector.forward_project(gt_volume, angles)
        recon = projector.fdk_reconstruct(projections, angles)
        metrics = evaluate_reconstruction(recon, gt_volume, data_range=1.0)

        view_dir = case_dir / f'{n_views}views'
        view_dir.mkdir(parents=True, exist_ok=True)

        save_numpy(projections, view_dir / 'projections.npy')
        save_numpy(angles.astype(np.float32), view_dir / 'angles.npy')
        save_numpy(recon.astype(np.float32), view_dir / 'fdk_recon.npy')
        (view_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

        summary['results'][str(n_views)] = {
            'projections': str(view_dir / 'projections.npy'),
            'angles': str(view_dir / 'angles.npy'),
            'fdk_recon': str(view_dir / 'fdk_recon.npy'),
            'metrics': metrics,
        }

    return summary
