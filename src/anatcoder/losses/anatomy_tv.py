"""Anatomy-aware total variation regularization."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from anatcoder.models.renderer import _query_model_density


def _voxel_ijk_to_normalized(
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    shape: tuple[int, int, int],
    volume_size_world: Sequence[float],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map voxel indices to model coordinates in normalized ``[0,1]^3``."""
    d, h, w = shape
    world = torch.as_tensor(volume_size_world, dtype=torch.float32, device=i.device)
    if world.shape != (3,):
        raise ValueError(f'volume_size_world must have 3 values, got shape={tuple(world.shape)}')
    if torch.any(world <= 0):
        raise ValueError(f'volume_size_world must be positive, got {list(volume_size_world)}')

    dz = world[0] / float(d)
    dy = world[1] / float(h)
    dx = world[2] / float(w)

    z = (i.to(torch.float32) - (float(d) - 1.0) * 0.5) * dz
    y = (j.to(torch.float32) - (float(h) - 1.0) * 0.5) * dy
    x = (k.to(torch.float32) - (float(w) - 1.0) * 0.5) * dx
    coords_world = torch.stack([z, y, x], dim=-1)
    coords_norm = coords_world / world + 0.5
    return coords_norm.to(dtype=dtype)


def anatomy_tv_loss(
    model: nn.Module,
    seg_volume: torch.Tensor,
    volume_size_world: Sequence[float],
    n_sample_pairs: int = 4096,
    alpha: float = 0.05,
    n_anatomy_classes: int = 0,
    device: torch.device | str = 'cuda',
) -> torch.Tensor:
    """Estimate anatomy-aware TV by random adjacent-voxel pairs.

    Uses nearest-neighbor class weights:
    - same class: weight=1.0
    - different class: weight=alpha
    """
    seg = torch.as_tensor(seg_volume, device=device)
    if seg.ndim != 3:
        raise ValueError(f'seg_volume must be [D,H,W], got shape={tuple(seg.shape)}')
    if n_sample_pairs <= 0:
        raise ValueError(f'n_sample_pairs must be positive, got {n_sample_pairs}')
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f'alpha must be in [0,1], got {alpha}')

    d, h, w = [int(v) for v in seg.shape]
    if d < 3 or h < 3 or w < 3:
        raise ValueError(f'seg_volume must have each dimension >=3, got shape={tuple(seg.shape)}')

    seg = seg.to(dtype=torch.long)
    try:
        query_dtype = next(model.parameters()).dtype
    except StopIteration:
        query_dtype = torch.float32

    i = torch.randint(1, d - 1, (n_sample_pairs,), device=seg.device)
    j = torch.randint(1, h - 1, (n_sample_pairs,), device=seg.device)
    k = torch.randint(1, w - 1, (n_sample_pairs,), device=seg.device)
    axis = torch.randint(0, 3, (n_sample_pairs,), device=seg.device)
    sign = torch.randint(0, 2, (n_sample_pairs,), device=seg.device, dtype=torch.long) * 2 - 1

    di = (axis == 0).to(torch.long) * sign
    dj = (axis == 1).to(torch.long) * sign
    dk = (axis == 2).to(torch.long) * sign

    i_n = i + di
    j_n = j + dj
    k_n = k + dk

    seg_center = seg[i, j, k]
    seg_neighbor = seg[i_n, j_n, k_n]
    if n_anatomy_classes > 0:
        seg_center = seg_center.clamp(0, int(n_anatomy_classes) - 1)
        seg_neighbor = seg_neighbor.clamp(0, int(n_anatomy_classes) - 1)

    same_class = (seg_center == seg_neighbor).to(dtype=query_dtype)
    weight = same_class + (1.0 - same_class) * float(alpha)

    coords_center = _voxel_ijk_to_normalized(
        i=i,
        j=j,
        k=k,
        shape=(d, h, w),
        volume_size_world=volume_size_world,
        dtype=query_dtype,
    )
    coords_neighbor = _voxel_ijk_to_normalized(
        i=i_n,
        j=j_n,
        k=k_n,
        shape=(d, h, w),
        volume_size_world=volume_size_world,
        dtype=query_dtype,
    )

    labels_center = seg_center if n_anatomy_classes > 0 else None
    labels_neighbor = seg_neighbor if n_anatomy_classes > 0 else None

    mu_center = _query_model_density(
        model=model,
        coords=coords_center,
        anatomy_labels=labels_center,
        n_anatomy_classes=int(max(0, n_anatomy_classes)),
    )
    mu_neighbor = _query_model_density(
        model=model,
        coords=coords_neighbor,
        anatomy_labels=labels_neighbor,
        n_anatomy_classes=int(max(0, n_anatomy_classes)),
    )
    tv = weight * (mu_center - mu_neighbor).abs().squeeze(-1)
    return tv.mean()

