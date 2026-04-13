"""Volume rendering utilities for Beer-Lambert CT projection synthesis."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anatcoder.models.ray_utils import sample_points_along_rays


class VolumeRenderer(nn.Module):
    """Integrate attenuation along rays with Beer-Lambert line-integral form."""

    def __init__(self) -> None:
        """Initialize renderer module."""
        super().__init__()

    def forward(self, densities: torch.Tensor, step_sizes: torch.Tensor) -> torch.Tensor:
        """Render line integrals from sampled densities and step lengths.

        Args:
            densities: Tensor ``[N_rays, N_samples, 1]`` or ``[N_rays, N_samples]``.
            step_sizes: Tensor ``[N_rays, N_samples]``.

        Returns:
            Line-integral values with shape ``[N_rays]``.
        """
        if densities.ndim == 3 and densities.shape[-1] == 1:
            densities = densities.squeeze(-1)
        if densities.ndim != 2:
            raise ValueError(f'densities must be [N,S] or [N,S,1], got {tuple(densities.shape)}')
        if step_sizes.ndim != 2:
            raise ValueError(f'step_sizes must be [N,S], got {tuple(step_sizes.shape)}')
        if densities.shape != step_sizes.shape:
            raise ValueError(f'shape mismatch: densities={densities.shape}, step_sizes={step_sizes.shape}')
        return torch.sum(densities * step_sizes, dim=1)


def _query_model_density(
    model: nn.Module,
    coords: torch.Tensor,
    anatomy_labels: torch.Tensor | None = None,
    n_anatomy_classes: int = 0,
) -> torch.Tensor:
    """Query model density with optional anatomy condition labels.

    Keeps backward compatibility for models that only accept positional inputs.
    """
    fn = model.query_density if hasattr(model, 'query_density') else model
    if anatomy_labels is None:
        return fn(coords)
    try:
        return fn(coords, anatomy_labels=anatomy_labels)
    except TypeError as exc:
        # Backward compatibility: older oracle path expects one-hot conditioning.
        if n_anatomy_classes <= 0:
            raise TypeError(
                'Model does not accept anatomy_labels and no valid class count was provided for one-hot fallback.'
            ) from exc
        labels = anatomy_labels.to(dtype=torch.long).clamp(0, int(n_anatomy_classes) - 1)
        anatomy_onehot = F.one_hot(labels, num_classes=int(n_anatomy_classes)).to(dtype=coords.dtype)
        try:
            return fn(coords, anatomy_onehot=anatomy_onehot)
        except TypeError as exc_onehot:
            raise TypeError(
                'Model does not accept anatomy_labels or anatomy_onehot, but anatomy conditioning was requested.'
            ) from exc_onehot


def _prepare_seg_volume_labels(
    seg_volume: torch.Tensor | np.ndarray | None,
    n_anatomy_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert segmentation input into float label volume ``[1,1,D,H,W]``."""
    if seg_volume is None:
        return None
    if n_anatomy_classes <= 0:
        raise ValueError(f'n_anatomy_classes must be positive when seg_volume is provided, got {n_anatomy_classes}')

    seg_t = torch.as_tensor(seg_volume)
    labels: torch.Tensor
    if seg_t.ndim == 3:
        labels = seg_t.to(dtype=torch.long)
    elif seg_t.ndim == 4:
        if seg_t.shape[0] == n_anatomy_classes:
            labels = torch.argmax(seg_t, dim=0)
        elif seg_t.shape[0] == 1:
            labels = seg_t.squeeze(0).to(dtype=torch.long)
        else:
            raise ValueError(
                '4D seg volume must be [C,D,H,W] with C=n_anatomy_classes or [1,D,H,W], '
                f'got shape={tuple(seg_t.shape)} and C={n_anatomy_classes}'
            )
    elif seg_t.ndim == 5:
        if seg_t.shape[0] != 1:
            raise ValueError(f'5D seg volume must have batch size 1, got shape={tuple(seg_t.shape)}')
        if seg_t.shape[1] == n_anatomy_classes:
            labels = torch.argmax(seg_t, dim=1).squeeze(0)
        elif seg_t.shape[1] == 1:
            labels = seg_t.squeeze(0).squeeze(0).to(dtype=torch.long)
        else:
            raise ValueError(
                '5D seg volume must be [1,C,D,H,W] with C=n_anatomy_classes or C=1, '
                f'got shape={tuple(seg_t.shape)} and C={n_anatomy_classes}'
            )
    else:
        raise ValueError(f'seg_volume must be 3D labels or 4D/5D one-hot/label, got shape={tuple(seg_t.shape)}')

    labels = labels.clamp(0, n_anatomy_classes - 1)
    return labels.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)


def _sample_anatomy_labels(
    seg_volume_labels: torch.Tensor,
    points_norm: torch.Tensor,
    n_anatomy_classes: int,
) -> torch.Tensor:
    """Sample anatomy label ids at normalized points.

    Args:
        seg_volume_labels: Float label volume ``[1,1,D,H,W]``.
        points_norm: Normalized coordinates in ``[0,1]^3`` with ``[..., 3]`` in ``(z,y,x)``.
        n_anatomy_classes: Number of anatomy classes.

    Returns:
        Label ids with shape ``[N]``.
    """
    flat_points = points_norm.reshape(-1, 3)
    # grid_sample expects xyz axis order in the last grid dimension.
    grid_xyz = flat_points[:, [2, 1, 0]] * 2.0 - 1.0
    grid_xyz = grid_xyz.reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        seg_volume_labels,
        grid_xyz,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True,
    )
    labels = sampled.squeeze(0).squeeze(0).squeeze(-1).squeeze(-1).reshape(-1)
    return labels.round().to(torch.long).clamp(0, int(n_anatomy_classes) - 1)


def _prepare_seg_volume_onehot(
    seg_volume: torch.Tensor | np.ndarray | None,
    n_anatomy_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Deprecated wrapper retained for compatibility with older tests/imports."""
    labels = _prepare_seg_volume_labels(seg_volume=seg_volume, n_anatomy_classes=n_anatomy_classes, device=device)
    if labels is None:
        return None
    labels_long = labels.squeeze(0).squeeze(0).to(dtype=torch.long)
    return F.one_hot(labels_long, num_classes=n_anatomy_classes).permute(3, 0, 1, 2).unsqueeze(0).to(
        device=device, dtype=dtype, non_blocking=True
    )


def _sample_anatomy_onehot(
    seg_volume_onehot: torch.Tensor,
    points_norm: torch.Tensor,
) -> torch.Tensor:
    """Deprecated wrapper retained for compatibility with older tests/imports."""
    n_classes = int(seg_volume_onehot.shape[1])
    labels = _sample_anatomy_labels(
        seg_volume_labels=_prepare_seg_volume_labels(
            seg_volume=seg_volume_onehot, n_anatomy_classes=n_classes, device=seg_volume_onehot.device
        ),
        points_norm=points_norm,
        n_anatomy_classes=n_classes,
    )
    return F.one_hot(labels, num_classes=n_classes).to(dtype=points_norm.dtype)


def render_rays(
    model: nn.Module,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    n_samples: int,
    near: float,
    far: float,
    perturb: bool = True,
    chunk_size: int = 4096,
    seg_volume: torch.Tensor | np.ndarray | None = None,
    n_anatomy_classes: int = 0,
    debug_capture: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Render projection values for a batch of rays.

    Args:
        model: INR model exposing ``query_density`` or ``forward``.
        ray_origins: Tensor ``[N, 3]``.
        ray_directions: Tensor ``[N, 3]``.
        n_samples: Number of points sampled per ray.
        near: Near sampling distance.
        far: Far sampling distance.
        perturb: Enable stratified random perturbation.
        chunk_size: Query chunk size to prevent OOM.
        seg_volume: Optional segmentation volume used to query anatomy labels.
        n_anatomy_classes: Number of anatomy classes for seg-label conditioning.

    Returns:
        Predicted line-integral tensor with shape ``[N]``.
    """
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    points, step_sizes = sample_points_along_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        n_samples=n_samples,
        near=near,
        far=far,
        perturb=perturb,
    )
    volume_size = getattr(model, 'volume_size_world', None)
    if volume_size is None:
        volume_size = getattr(model, 'volume_size_mm', None)
    if volume_size is None:
        raise AttributeError('model must define `volume_size_world` (or `volume_size_mm`) for normalization')
    volume_size_t = torch.as_tensor(volume_size, dtype=points.dtype, device=points.device)
    if torch.any(volume_size_t <= 0):
        raise ValueError(f'volume size must be positive, got {volume_size}')
    points_norm = points / volume_size_t + 0.5

    inside_mask: torch.Tensor | None = None
    if bool(getattr(model, 'zero_outside_volume', False)):
        inside_mask = ((points_norm >= 0.0) & (points_norm <= 1.0)).all(dim=-1)
        points_query = points_norm.clamp(0.0, 1.0)
    else:
        points_query = points_norm

    flat_points = points_query.reshape(-1, 3)
    if debug_capture is not None:
        seg_sample_points = points_query.reshape(-1, 3)
        debug_capture['seg_sample_coords_preview'] = seg_sample_points[:5].detach().cpu()
        debug_capture['coord_same_storage'] = (
            seg_sample_points.untyped_storage().data_ptr() == flat_points.untyped_storage().data_ptr()
        )
        if seg_sample_points.numel() > 0:
            debug_capture['coord_max_abs_diff'] = float((seg_sample_points - flat_points).abs().max().item())
        else:
            debug_capture['coord_max_abs_diff'] = 0.0
    flat_anatomy_labels: torch.Tensor | None = None
    if seg_volume is not None:
        seg_volume_labels = _prepare_seg_volume_labels(
            seg_volume=seg_volume,
            n_anatomy_classes=int(n_anatomy_classes),
            device=flat_points.device,
        )
        if seg_volume_labels is None:
            raise RuntimeError('seg_volume preparation failed unexpectedly')
        flat_anatomy_labels = _sample_anatomy_labels(
            seg_volume_labels=seg_volume_labels,
            points_norm=points_query,
            n_anatomy_classes=int(n_anatomy_classes),
        )
        if debug_capture is not None:
            debug_capture['anatomy_labels'] = flat_anatomy_labels

    pred_chunks: list[torch.Tensor] = []
    for start in range(0, flat_points.shape[0], chunk_size):
        end = min(start + chunk_size, flat_points.shape[0])
        query = flat_points[start:end]
        if debug_capture is not None and 'network_query_coords_preview' not in debug_capture:
            debug_capture['network_query_coords_preview'] = query[:5].detach().cpu()
        cond_chunk = flat_anatomy_labels[start:end] if flat_anatomy_labels is not None else None
        pred = _query_model_density(
            model,
            query,
            anatomy_labels=cond_chunk,
            n_anatomy_classes=int(n_anatomy_classes),
        )
        pred_chunks.append(pred)

    densities = torch.cat(pred_chunks, dim=0).reshape(ray_origins.shape[0], n_samples, 1)
    if inside_mask is not None:
        densities = densities * inside_mask.to(dtype=densities.dtype).unsqueeze(-1)
    line_integral_scale = float(getattr(model, 'line_integral_scale', 1.0))
    if line_integral_scale <= 0:
        raise ValueError(f'model.line_integral_scale must be positive, got {line_integral_scale}')
    step_sizes = step_sizes * line_integral_scale
    renderer = VolumeRenderer().to(ray_origins.device)
    return renderer(densities, step_sizes)


def reconstruct_volume(
    model: nn.Module,
    volume_size: list[int],
    voxel_size: list[float],
    chunk_size: int = 65536,
    device: torch.device = torch.device('cuda'),
    seg_volume: torch.Tensor | np.ndarray | None = None,
    n_anatomy_classes: int = 0,
) -> np.ndarray:
    """Reconstruct a dense 3D volume from an implicit density model."""
    if len(volume_size) != 3:
        raise ValueError(f'volume_size must have length 3, got {volume_size}')
    if len(voxel_size) != 3:
        raise ValueError(f'voxel_size must have length 3, got {voxel_size}')
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    nz, ny, nx = [int(v) for v in volume_size]
    voxel_size_world = getattr(model, 'voxel_size_world', voxel_size)
    if len(voxel_size_world) != 3:
        raise ValueError(f'voxel_size_world must have length 3, got {voxel_size_world}')
    dz, dy, dx = [float(v) for v in voxel_size_world]
    volume_size_world = getattr(model, 'volume_size_world', None)
    if volume_size_world is None:
        volume_size_world = getattr(model, 'volume_size_mm', None)
    if volume_size_world is None:
        volume_size_world = [float(nz * dz), float(ny * dy), float(nx * dx)]
    if len(volume_size_world) != 3:
        raise ValueError(f'volume_size_world must have length 3, got {volume_size_world}')
    batch_size = int(chunk_size)

    x = torch.linspace(-(nx - 1) / 2 * dx, (nx - 1) / 2 * dx, nx)
    y = torch.linspace(-(ny - 1) / 2 * dy, (ny - 1) / 2 * dy, ny)
    z = torch.linspace(-(nz - 1) / 2 * dz, (nz - 1) / 2 * dz, nz)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    phys = torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)
    volume_size_t = torch.as_tensor(volume_size_world, dtype=phys.dtype, device=phys.device)
    if torch.any(volume_size_t <= 0):
        raise ValueError(f'volume size must be positive, got {volume_size_world}')
    normalized = phys / volume_size_t + 0.5
    if bool(getattr(model, 'zero_outside_volume', False)):
        normalized = normalized.clamp(0.0, 1.0)

    try:
        infer_device = next(model.parameters()).device
    except StopIteration:
        infer_device = device
    device = infer_device
    model.eval()
    mu_all: list[torch.Tensor] = []
    seg_volume_labels: torch.Tensor | None = None
    if seg_volume is not None:
        seg_volume_labels = _prepare_seg_volume_labels(
            seg_volume=seg_volume,
            n_anatomy_classes=int(n_anatomy_classes),
            device=device,
        )
        if seg_volume_labels is None:
            raise RuntimeError('seg_volume preparation failed unexpectedly')

    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            end = min(start + batch_size, normalized.shape[0])
            chunk = normalized[start:end].to(device)
            cond_chunk: torch.Tensor | None = None
            if seg_volume_labels is not None:
                cond_chunk = _sample_anatomy_labels(
                    seg_volume_labels=seg_volume_labels,
                    points_norm=chunk,
                    n_anatomy_classes=int(n_anatomy_classes),
                )
            mu = _query_model_density(
                model,
                chunk,
                anatomy_labels=cond_chunk,
                n_anatomy_classes=int(n_anatomy_classes),
            )
            mu_all.append(mu.cpu())

    mu_all = torch.cat(mu_all, dim=0)
    volume = mu_all.squeeze(-1).reshape(nz, ny, nx).numpy()
    return volume.astype(np.float32, copy=False)
