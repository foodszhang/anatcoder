"""Dataset and datamodule for per-scan sparse-view INR CT reconstruction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from anatcoder.models.ray_utils import generate_rays_for_view
from anatcoder.utils.geometry import CBCTGeometry
from anatcoder.utils.io import load_numpy


class CTProjectionDataset(Dataset[dict[str, torch.Tensor]]):
    """Projection-ray dataset for one CT case and one sparse-view setting."""

    def __init__(
        self,
        case_dir: str,
        proj_dir: str,
        n_views: int,
        geo: CBCTGeometry,
    ) -> None:
        """Load projections, angles, GT volume and precompute per-pixel rays."""
        super().__init__()
        self.case_dir = Path(case_dir)
        self.proj_case_dir = Path(proj_dir)
        self.n_views = int(n_views)
        self.geo = geo

        view_dir = self.proj_case_dir / f'{self.n_views}views'
        proj_path = view_dir / 'projections.npy'
        angle_path = view_dir / 'angles.npy'
        volume_path = self.case_dir / 'volume.npy'
        seg_path = self.case_dir / 'seg.npy'

        if not proj_path.exists():
            raise FileNotFoundError(f'Projection file not found: {proj_path}')
        if not angle_path.exists():
            raise FileNotFoundError(f'Angles file not found: {angle_path}')
        if not volume_path.exists():
            raise FileNotFoundError(f'GT volume not found: {volume_path}')

        self.projections = np.asarray(load_numpy(proj_path), dtype=np.float32)
        self.angles = np.asarray(load_numpy(angle_path), dtype=np.float32)
        self.gt_volume = np.asarray(load_numpy(volume_path), dtype=np.float32)
        self.seg = np.asarray(load_numpy(seg_path), dtype=np.int16) if seg_path.exists() else None

        if self.projections.ndim != 3:
            raise ValueError(f'projections must be [K,H,W], got {self.projections.shape}')
        if self.angles.ndim != 1:
            raise ValueError(f'angles must be [K], got {self.angles.shape}')
        if self.projections.shape[0] != self.angles.shape[0]:
            raise ValueError(
                f'Projection/angle mismatch: {self.projections.shape[0]} vs {self.angles.shape[0]}'
            )

        self.n_view_loaded, self.det_rows, self.det_cols = self.projections.shape
        self.volume_size_mm = (np.asarray(self.geo.n_voxel, dtype=np.float32) * np.asarray(self.geo.d_voxel, dtype=np.float32))
        diag = float(np.linalg.norm(self.volume_size_mm))
        self.near = float(self.geo.DSO - 0.5 * diag)
        self.far = float(self.geo.DSO + 0.5 * diag)

        # Precompute all rays once for fast __getitem__.
        ray_origins: list[np.ndarray] = []
        ray_directions: list[np.ndarray] = []
        for angle in self.angles:
            origins_t, dirs_t = generate_rays_for_view(
                self.geo,
                float(angle),
                device=torch.device('cpu'),
            )
            ray_origins.append(origins_t.numpy())
            ray_directions.append(dirs_t.numpy())
        self._ray_origins = np.stack(ray_origins, axis=0).reshape(-1, 3).astype(np.float32, copy=False)
        self._ray_directions = np.stack(ray_directions, axis=0).reshape(-1, 3).astype(np.float32, copy=False)
        self._gt_pixels = self.projections.reshape(-1).astype(np.float32, copy=False)

    def __len__(self) -> int:
        """Return total number of rays: ``K * detector_rows * detector_cols``."""
        return int(self._gt_pixels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one ray sample and corresponding GT projection pixel value."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Index out of range: {idx}')

        return {
            'ray_origin': torch.from_numpy(self._ray_origins[idx]),
            'ray_direction': torch.from_numpy(self._ray_directions[idx]),
            'gt_pixel': torch.tensor(self._gt_pixels[idx], dtype=torch.float32),
        }


class RayBatchSampler(Sampler[list[int]]):
    """Batch sampler for flattened ray indices."""

    def __init__(self, num_rays: int, batch_size: int, shuffle: bool = True) -> None:
        """Initialize ray batch sampler."""
        self.num_rays = int(num_rays)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        if self.num_rays <= 0:
            raise ValueError(f'num_rays must be positive, got {num_rays}')
        if self.batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')

    def __iter__(self):
        """Yield batches of ray indices."""
        if self.shuffle:
            indices = torch.randperm(self.num_rays).tolist()
        else:
            indices = list(range(self.num_rays))
        for start in range(0, self.num_rays, self.batch_size):
            yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        """Return number of batches per pass."""
        return math.ceil(self.num_rays / self.batch_size)


class CTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for one-case INR optimization."""

    def __init__(
        self,
        case_dir: str,
        proj_dir: str,
        n_views: int,
        geo: CBCTGeometry,
        batch_size: int = 4096,
        num_workers: int = 4,
    ) -> None:
        """Store dataset construction arguments."""
        super().__init__()
        self.case_dir = case_dir
        self.proj_dir = proj_dir
        self.n_views = int(n_views)
        self.geo = geo
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.dataset: CTProjectionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create projection dataset."""
        _ = stage
        self.dataset = CTProjectionDataset(
            case_dir=self.case_dir,
            proj_dir=self.proj_dir,
            n_views=self.n_views,
            geo=self.geo,
        )

    def train_dataloader(self) -> DataLoader:
        """Return shuffled ray dataloader for training."""
        if self.dataset is None:
            raise RuntimeError('DataModule.setup() must be called before train_dataloader().')
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a lightweight loader used to trigger validation hooks."""
        if self.dataset is None:
            raise RuntimeError('DataModule.setup() must be called before val_dataloader().')
        # Keep val loader cheap; full-volume validation happens in module.
        subset_len = min(len(self.dataset), max(1, self.batch_size))
        indices = torch.arange(subset_len, dtype=torch.long).tolist()
        return DataLoader(
            torch.utils.data.Subset(self.dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

    def get_gt_volume(self) -> np.ndarray:
        """Return ground-truth volume for metric computation."""
        if self.dataset is None:
            raise RuntimeError('DataModule.setup() must be called before get_gt_volume().')
        return self.dataset.gt_volume

    def get_geo(self) -> CBCTGeometry:
        """Return geometry object."""
        return self.geo


class CTReconDataset(CTProjectionDataset):
    """Backward-compatible alias class for historical dataset naming."""

    def __init__(
        self,
        data_dir: str | Path,
        proj_dir: str | Path,
        split: str = 'train',
        n_samples_per_ray: int = 192,
    ) -> None:
        """Adapt legacy ctor arguments to Week-2 dataset implementation."""
        _ = split
        _ = n_samples_per_ray
        case_dir = Path(data_dir)
        proj_case_dir = Path(proj_dir)
        super().__init__(
            case_dir=str(case_dir),
            proj_dir=str(proj_case_dir),
            n_views=50,
            geo=CBCTGeometry(),
        )
