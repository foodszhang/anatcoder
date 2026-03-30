"""Classical FDK/SART baseline wrappers based on TIGRE."""

from typing import Any

import numpy as np


class FDKBaseline:
    """Classical baseline runner for FDK and SART reconstruction."""

    def __init__(self, geo_params: dict[str, Any]) -> None:
        """Store reconstruction geometry parameters."""
        self.geo_params = geo_params

    def reconstruct_fdk(self, projections: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Run FDK reconstruction from projection data."""
        raise NotImplementedError("TODO: implement FDK baseline")

    def reconstruct_sart(self, projections: np.ndarray, angles: np.ndarray, n_iter: int = 50) -> np.ndarray:
        """Run SART reconstruction from projection data."""
        raise NotImplementedError("TODO: implement SART baseline")
