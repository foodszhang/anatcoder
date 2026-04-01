"""Volume rendering utilities for Beer-Lambert CT projection synthesis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from anatcoder.models.ray_utils import _to_tigre_world, normalize_coords, sample_points_along_rays


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
    volume_size = getattr(model, 'volume_size_mm', None)
    if volume_size is None:
        raise AttributeError('model must define `volume_size_mm` for coordinate normalization')
    points_norm = normalize_coords(points, list(volume_size))

    flat_points = points_norm.reshape(-1, 3)
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
    renderer = VolumeRenderer().to(ray_origins.device)
    return renderer(densities, step_sizes)


def reconstruct_volume(
    model: nn.Module,
    volume_size: list[int],
    voxel_size: list[float],
    chunk_size: int = 65536,
    device: torch.device = torch.device('cuda'),
) -> np.ndarray:
    """Query model density on full voxel grid and reconstruct 3D volume.

    Args:
        model: Trained INR model.
        volume_size: Integer shape ``[Nz, Ny, Nx]``.
        voxel_size: Physical spacing ``[dz, dy, dx]`` in millimeters.
        chunk_size: Chunk size for model querying.
        device: Computation device.

    Returns:
        Reconstructed volume as ``float32`` NumPy array with shape ``[Nz, Ny, Nx]``.
    """
    if len(volume_size) != 3:
        raise ValueError(f'volume_size must have length 3, got {volume_size}')
    if len(voxel_size) != 3:
        raise ValueError(f'voxel_size must have length 3, got {voxel_size}')
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    nz, ny, nx = [int(v) for v in volume_size]
    dz, dy, dx = [float(v) for v in voxel_size]

    z = (torch.arange(nz, device=device, dtype=torch.float32) + 0.5 - nz / 2.0) * dz
    y = (torch.arange(ny, device=device, dtype=torch.float32) + 0.5 - ny / 2.0) * dy
    x = (torch.arange(nx, device=device, dtype=torch.float32) + 0.5 - nx / 2.0) * dx
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    world_points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    world_points = _to_tigre_world(world_points)

    size_mm = [nz * dz, ny * dy, nx * dx]
    norm_points = normalize_coords(world_points, size_mm)

    model = model.to(device)
    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, norm_points.shape[0], chunk_size):
            end = min(start + chunk_size, norm_points.shape[0])
            query = norm_points[start:end]
            if hasattr(model, 'query_density'):
                out = model.query_density(query)
            else:
                out = model(query)
            preds.append(out)

    mu = torch.cat(preds, dim=0).reshape(nz, ny, nx)
    return mu.detach().cpu().numpy().astype(np.float32, copy=False)
