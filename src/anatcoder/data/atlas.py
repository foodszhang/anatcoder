"""Atlas construction and query interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


class AtlasBuilder:
    """Build atlas probability maps from segmentation sources."""

    @staticmethod
    def from_oracle(seg_path: str | Path, n_classes: int = 105) -> np.ndarray:
        """Build an Oracle atlas from one segmentation map.

        Args:
            seg_path: Path to ``seg.npy`` with integer labels.
            n_classes: Number of classes used for one-hot conversion.

        Returns:
            Atlas probability volume of shape ``[n_classes, D, H, W]``.
        """
        seg_file = Path(seg_path)
        if not seg_file.exists():
            raise FileNotFoundError(f'Segmentation file not found: {seg_file}')
        if n_classes <= 1:
            raise ValueError(f'n_classes must be > 1, got: {n_classes}')

        seg = np.asarray(np.load(seg_file), dtype=np.int64)
        if seg.ndim != 3:
            raise ValueError(f'Segmentation must be 3D, got shape: {seg.shape}')
        if seg.min() < 0:
            raise ValueError('Segmentation contains negative labels')
        if seg.max(initial=0) >= n_classes:
            raise ValueError(
                f'Segmentation label max ({int(seg.max())}) exceeds n_classes-1 ({n_classes - 1})'
            )

        atlas = np.zeros((n_classes, *seg.shape), dtype=np.float32)
        for class_id in np.unique(seg):
            atlas[int(class_id), ...] = (seg == class_id).astype(np.float32)
        return atlas

    @staticmethod
    def from_population(
        seg_paths: Sequence[str | Path],
        reference_path: str | Path,
        n_cases: int = 50,
    ) -> Tensor:
        """Build a statistical atlas from multiple cases (Week 3-4)."""
        _ = (seg_paths, reference_path, n_cases)
        raise NotImplementedError('Population atlas: Week 3-4')


class AtlasQuerier:
    """Query atlas class probabilities at normalized coordinates."""

    def __init__(self, atlas_path: str | Path) -> None:
        """Load an atlas ``.npy`` array into memory.

        Args:
            atlas_path: Path to atlas array with shape ``[C, D, H, W]``.
        """
        atlas_file = Path(atlas_path)
        if not atlas_file.exists():
            raise FileNotFoundError(f'Atlas file not found: {atlas_file}')

        atlas_np = np.asarray(np.load(atlas_file), dtype=np.float32)
        if atlas_np.ndim != 4:
            raise ValueError(f'Atlas must be 4D [C, D, H, W], got shape: {atlas_np.shape}')

        self.atlas_path = atlas_file
        self.atlas = torch.from_numpy(atlas_np)
        self.n_classes = int(atlas_np.shape[0])

    def query(self, coords: Tensor) -> Tensor:
        """Query atlas probabilities via trilinear interpolation.

        Args:
            coords: Normalized coordinates ``[N, 3]`` in ``[0, 1]``.
                Coordinate order is ``[z, y, x]``.

        Returns:
            Probability vectors ``[N, n_classes]``.
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f'coords must have shape [N, 3], got: {tuple(coords.shape)}')

        device = coords.device
        coords_clamped = coords.to(dtype=torch.float32).clamp(0.0, 1.0)

        atlas = self.atlas.to(device=device, dtype=torch.float32).unsqueeze(0)

        # grid_sample expects grid order [x, y, z] in [-1, 1].
        grid_x = coords_clamped[:, 2] * 2.0 - 1.0
        grid_y = coords_clamped[:, 1] * 2.0 - 1.0
        grid_z = coords_clamped[:, 0] * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, -1, 1, 1, 3)

        sampled = F.grid_sample(
            atlas,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        probs = sampled.squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)

        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return probs
