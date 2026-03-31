"""Geometry helper functions for CBCT setup and angle generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CBCTGeometry:
    """CBCT scanning geometry parameters.

    Attributes:
        DSD: Source-to-detector distance in millimeters.
        DSO: Source-to-object distance in millimeters.
        n_voxel: Volume size as ``[Nz, Ny, Nx]``.
        d_voxel: Voxel spacing in millimeters as ``[dz, dy, dx]``.
        n_detector: Detector resolution as ``[rows, cols]``.
        d_detector: Detector pixel spacing in millimeters as ``[drow, dcol]``.
    """

    DSD: float = 1536.0
    DSO: float = 1000.0
    n_voxel: list[int] | None = None
    d_voxel: list[float] | None = None
    n_detector: list[int] | None = None
    d_detector: list[float] | None = None

    def __post_init__(self) -> None:
        """Fill defaults and validate geometry arrays."""
        if self.n_voxel is None:
            self.n_voxel = [128, 128, 128]
        if self.d_voxel is None:
            self.d_voxel = [1.0, 1.0, 1.0]
        if self.n_detector is None:
            self.n_detector = [256, 256]
        if self.d_detector is None:
            self.d_detector = [1.5, 1.5]

        if len(self.n_voxel) != 3 or len(self.d_voxel) != 3:
            raise ValueError('n_voxel and d_voxel must have length 3')
        if len(self.n_detector) != 2 or len(self.d_detector) != 2:
            raise ValueError('n_detector and d_detector must have length 2')

    def to_tigre_geometry(self):
        """Convert this dataclass into a TIGRE ``geometry`` object."""
        try:
            import tigre
        except ImportError as exc:
            raise ImportError(
                "TIGRE is required for projection/reconstruction. Install with `uv pip install tigre`."
            ) from exc

        geo = tigre.geometry(mode='cone')
        geo.DSD = float(self.DSD)
        geo.DSO = float(self.DSO)

        geo.nVoxel = np.asarray(self.n_voxel, dtype=np.int32)
        geo.dVoxel = np.asarray(self.d_voxel, dtype=np.float32)
        geo.sVoxel = geo.nVoxel.astype(np.float32) * geo.dVoxel

        geo.nDetector = np.asarray(self.n_detector, dtype=np.int32)
        geo.dDetector = np.asarray(self.d_detector, dtype=np.float32)
        geo.sDetector = geo.nDetector.astype(np.float32) * geo.dDetector

        geo.offOrigin = np.zeros(3, dtype=np.float32)
        geo.offDetector = np.zeros(2, dtype=np.float32)
        geo.accuracy = 0.5
        geo.rotDetector = np.zeros(3, dtype=np.float32)
        return geo


def generate_angles(
    n_views: int,
    start: float = 0.0,
    end: float = 360.0,
    endpoint: bool = False,
) -> np.ndarray:
    """Generate uniformly sampled scan angles in radians.

    Args:
        n_views: Number of views.
        start: Start angle in degrees.
        end: End angle in degrees.
        endpoint: Whether to include ``end`` angle.

    Returns:
        NumPy array of angles in radians.
    """
    if n_views <= 0:
        raise ValueError(f'n_views must be positive, got: {n_views}')
    angles_deg = np.linspace(start, end, n_views, endpoint=endpoint, dtype=np.float32)
    return np.deg2rad(angles_deg).astype(np.float32)


def make_default_geo_params(
    dsd: float = 1536.0,
    dso: float = 1000.0,
    detector_size: tuple[int, int] = (256, 256),
    detector_spacing: tuple[float, float] = (1.5, 1.5),
) -> dict[str, object]:
    """Create a default geometry dictionary for compatibility with existing configs."""
    return {
        'DSD': dsd,
        'DSO': dso,
        'n_voxel': [128, 128, 128],
        'd_voxel': [1.0, 1.0, 1.0],
        'n_detector': list(detector_size),
        'd_detector': list(detector_spacing),
    }


def generate_uniform_angles(n_views: int, endpoint: bool = False) -> np.ndarray:
    """Backward-compatible wrapper around :func:`generate_angles`."""
    return generate_angles(n_views=n_views, start=0.0, end=360.0, endpoint=endpoint)
