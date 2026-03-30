"""Geometry helper functions for CBCT setup and angle generation."""

from collections.abc import Sequence

import numpy as np


def make_default_geo_params(
    dsd: float = 1536.0,
    dso: float = 1000.0,
    detector_size: Sequence[int] = (256, 256),
    detector_spacing: Sequence[float] = (1.5, 1.5),
) -> dict[str, object]:
    """Create a default CBCT geometry dictionary."""
    return {
        "DSD": dsd,
        "DSO": dso,
        "detector_size": list(detector_size),
        "detector_spacing": list(detector_spacing),
    }


def generate_uniform_angles(n_views: int, endpoint: bool = False) -> np.ndarray:
    """Generate uniformly distributed gantry angles in radians."""
    raise NotImplementedError("TODO: implement angle generation")
