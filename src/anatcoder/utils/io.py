"""I/O helpers for NIfTI, NumPy, and YAML serialization."""

from pathlib import Path
from typing import Any

import numpy as np


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume into a NumPy array."""
    raise NotImplementedError("TODO: implement NIfTI loading")


def save_nifti(array: np.ndarray, path: str | Path, reference_path: str | Path | None = None) -> None:
    """Save a NumPy array as NIfTI, optionally copying reference metadata."""
    raise NotImplementedError("TODO: implement NIfTI saving")


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk."""
    raise NotImplementedError("TODO: implement NPY loading")


def save_npy(array: np.ndarray, path: str | Path) -> None:
    """Persist a NumPy array to disk."""
    raise NotImplementedError("TODO: implement NPY saving")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML content into a Python dictionary."""
    raise NotImplementedError("TODO: implement YAML loading")
