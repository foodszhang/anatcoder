"""Volume rendering utilities for Beer-Lambert CT projection synthesis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

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


def render_rays(
    model: nn.Module,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    n_samples: int,
    near: float,
    far: float,
    perturb: bool = True,
    chunk_size: int = 4096,
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
    pred_chunks: list[torch.Tensor] = []
    for start in range(0, flat_points.shape[0], chunk_size):
        end = min(start + chunk_size, flat_points.shape[0])
        query = flat_points[start:end]
        if hasattr(model, 'query_density'):
            pred = model.query_density(query)
        else:
            pred = model(query)
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

    axis_mode = str(getattr(model, 'coord_axis_mode', 'legacy')).lower()
    x = torch.linspace(-(nx - 1) / 2 * dx, (nx - 1) / 2 * dx, nx)
    y = torch.linspace(-(ny - 1) / 2 * dy, (ny - 1) / 2 * dy, ny)
    z = torch.linspace(-(nz - 1) / 2 * dz, (nz - 1) / 2 * dz, nz)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    if axis_mode == 'identity':
        phys = torch.stack((zz, yy, xx), dim=-1).reshape(-1, 3)
    else:
        phys = torch.stack((-yy, -xx, -zz), dim=-1).reshape(-1, 3)
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
    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            end = min(start + batch_size, normalized.shape[0])
            chunk = normalized[start:end].to(device)
            if hasattr(model, 'query_density'):
                mu = model.query_density(chunk)
            else:
                mu = model(chunk)
            mu_all.append(mu.cpu())

    mu_all = torch.cat(mu_all, dim=0)
    volume = mu_all.squeeze(-1).reshape(nz, ny, nx).numpy()
    return volume.astype(np.float32, copy=False)
