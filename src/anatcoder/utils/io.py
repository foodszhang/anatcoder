"""I/O helpers for NIfTI, NumPy, and YAML serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import yaml


def load_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, tuple[float, ...]]]:
    """Load a NIfTI file.

    Args:
        path: Input NIfTI path.

    Returns:
        A tuple ``(volume, metadata)`` where ``volume`` is a ``[z, y, x]`` NumPy array
        and ``metadata`` contains ``spacing``, ``origin``, and ``direction``.
    """
    nifti_path = Path(path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    image = sitk.ReadImage(str(nifti_path))
    volume = sitk.GetArrayFromImage(image).copy()
    metadata = {
        'spacing': tuple(float(v) for v in image.GetSpacing()),
        'origin': tuple(float(v) for v in image.GetOrigin()),
        'direction': tuple(float(v) for v in image.GetDirection()),
    }
    return volume, metadata


def save_nifti(
    volume: np.ndarray,
    path: str | Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Save a NumPy volume as NIfTI.

    Args:
        volume: Input NumPy volume in ``[z, y, x]`` layout.
        path: Output ``.nii`` or ``.nii.gz`` path.
        spacing: Physical voxel spacing in millimeters.
        origin: Physical origin in millimeters.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.GetImageFromArray(np.asarray(volume))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetOrigin(tuple(float(v) for v in origin))
    sitk.WriteImage(image, str(out_path))


def load_numpy(path: str | Path) -> np.ndarray:
    """Load an ``.npy`` array from disk."""
    npy_path = Path(path)
    if not npy_path.exists():
        raise FileNotFoundError(f"NumPy file not found: {npy_path}")
    return np.load(npy_path)


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save an array as ``.npy`` and create parent directories automatically."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)


def load_npy(path: str | Path) -> np.ndarray:
    """Backward-compatible alias for :func:`load_numpy`."""
    return load_numpy(path)


def save_npy(array: np.ndarray, path: str | Path) -> None:
    """Backward-compatible alias for :func:`save_numpy`."""
    save_numpy(array, path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML content into a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed dictionary.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open('r', encoding='utf-8') as handle:
        content = yaml.safe_load(handle) or {}

    if not isinstance(content, dict):
        raise ValueError(f"YAML content must be a mapping: {yaml_path}")
    return content
