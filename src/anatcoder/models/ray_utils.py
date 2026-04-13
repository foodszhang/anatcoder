"""Ray generation and point sampling utilities for CBCT geometry."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from anatcoder.utils.geometry import CBCTGeometry


def _to_tigre_world(coords: torch.Tensor) -> torch.Tensor:
    """[Deprecated] Map legacy internal coordinates to TIGRE-compatible world axes.

    TIGRE's world frame for the Python volume convention can be matched by
    reordering/signing axes as ``[x, -z, -y]``.

    Deprecated: use :func:`_naf_to_tigre_world` via
    :func:`generate_rays_for_view_naf_tigre` for the default pipeline.
    """
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f'coords must be [N,3], got shape={tuple(coords.shape)}')
    return torch.stack((coords[:, 0], -coords[:, 2], -coords[:, 1]), dim=-1)


def _naf_to_tigre_world(coords: torch.Tensor) -> torch.Tensor:
    """Map raw NAF ray frame into TIGRE-aligned world axes."""
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f'coords must be [N,3], got shape={tuple(coords.shape)}')
    return torch.stack((-coords[:, 2], coords[:, 1], coords[:, 0]), dim=-1)


def _angle2pose(DSO_m: float, angle: float) -> np.ndarray:
    """NAF official-style pose transform from angle to camera-to-world matrix."""
    phi1 = -np.pi / 2
    R1 = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(phi1), -np.sin(phi1)], [0.0, np.sin(phi1), np.cos(phi1)]],
        dtype=np.float32,
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [[np.cos(phi2), -np.sin(phi2), 0.0], [np.sin(phi2), np.cos(phi2), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    R3 = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rot = R3 @ R2 @ R1
    trans = np.array([DSO_m * np.cos(angle), DSO_m * np.sin(angle), 0.0], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:-1, :-1] = rot
    T[:-1, -1] = trans
    return T


def generate_rays_for_view_naf(
    geo: CBCTGeometry,
    angle: float,
    device: torch.device | str = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one-view rays with NAF official convention (meter units)."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    DSD_m = float(geo.DSD) / 1000.0
    DSO_m = float(geo.DSO) / 1000.0
    d_det_u = float(geo.d_detector[0]) / 1000.0
    d_det_v = float(geo.d_detector[1]) / 1000.0

    width = int(geo.n_detector[1])
    height = int(geo.n_detector[0])

    pose = torch.tensor(_angle2pose(DSO_m, float(angle)), dtype=torch.float32, device=dev)
    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=dev, dtype=torch.float32),
        torch.linspace(0, height - 1, height, device=dev, dtype=torch.float32),
        indexing='ij',
    )
    uu = (i.t() + 0.5 - width / 2.0) * d_det_u
    vv = (j.t() + 0.5 - height / 2.0) * d_det_v
    dirs = torch.stack((uu / DSD_m, vv / DSD_m, torch.ones_like(uu)), dim=-1)

    rays_d = torch.matmul(pose[:3, :3], dirs.unsqueeze(-1)).squeeze(-1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = pose[:3, -1].expand_as(rays_d)
    return rays_o.reshape(-1, 3).to(torch.float32), rays_d.reshape(-1, 3).to(torch.float32)


def generate_rays_for_view_naf_tigre(
    geo: CBCTGeometry,
    angle: float,
    device: torch.device | str = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate NAF rays transformed into TIGRE-aligned world coordinates."""
    origins, directions = generate_rays_for_view_naf(geo=geo, angle=angle, device=device)
    origins = _naf_to_tigre_world(origins)
    directions = F.normalize(_naf_to_tigre_world(directions), dim=-1)
    return origins.to(torch.float32), directions.to(torch.float32)


def generate_rays_for_view(
    geo: CBCTGeometry,
    angle: float,
    device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    """[Deprecated] Generate legacy detector rays for one CBCT view angle.

    Args:
        geo: CBCT scanning geometry.
        angle: View angle in radians.
        device: Target torch device.

    Returns:
        A tuple ``(ray_origins, ray_directions)``:
            - ``ray_origins`` has shape ``[rows * cols, 3]``.
            - ``ray_directions`` has shape ``[rows * cols, 3]`` and unit norm.

    Deprecated: use :func:`generate_rays_for_view_naf_tigre`.
    """
    rows, cols = int(geo.n_detector[0]), int(geo.n_detector[1])
    d_row, d_col = float(geo.d_detector[0]), float(geo.d_detector[1])

    angle_tensor = torch.tensor(float(angle), dtype=torch.float32, device=device)
    cos_a = torch.cos(angle_tensor)
    sin_a = torch.sin(angle_tensor)

    source = torch.tensor(
        [-geo.DSO * sin_a, geo.DSO * cos_a, 0.0], dtype=torch.float32, device=device
    )

    center_radius = float(geo.DSO - geo.DSD)
    detector_center = torch.tensor(
        [-center_radius * sin_a, center_radius * cos_a, 0.0],
        dtype=torch.float32,
        device=device,
    )

    detector_u = torch.tensor([cos_a, sin_a, 0.0], dtype=torch.float32, device=device)
    detector_v = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    row_ids = torch.arange(rows, device=device, dtype=torch.float32)
    col_ids = torch.arange(cols, device=device, dtype=torch.float32)
    row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing='ij')

    row_offsets = ((rows - 1.0) * 0.5 - row_grid) * d_row
    col_offsets = (col_grid - (cols - 1.0) * 0.5) * d_col

    detector_points = (
        detector_center[None, None, :]
        + col_offsets[..., None] * detector_u[None, None, :]
        + row_offsets[..., None] * detector_v[None, None, :]
    )
    origins = source[None, None, :].expand(rows, cols, 3).reshape(-1, 3)
    directions = F.normalize((detector_points - source[None, None, :]).reshape(-1, 3), dim=-1)
    origins = _to_tigre_world(origins)
    directions = F.normalize(_to_tigre_world(directions), dim=-1)
    return origins.to(torch.float32), directions.to(torch.float32)


def generate_rays_batch(
    geo: CBCTGeometry,
    angles: torch.Tensor,
    n_rays: int,
    device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly sample rays from all detector pixels across all input views.

    Args:
        geo: CBCT scanning geometry.
        angles: Angle tensor with shape ``[K]`` in radians.
        n_rays: Number of rays to sample.
        device: Target torch device.

    Returns:
        A tuple ``(ray_origins, ray_directions, gt_pixels)``:
            - ``ray_origins`` shape ``[n_rays, 3]``.
            - ``ray_directions`` shape ``[n_rays, 3]``.
            - ``gt_pixels`` shape ``[n_rays]`` as flattened projection indices.
    """
    if angles.ndim != 1:
        raise ValueError(f'angles must be 1D, got shape={tuple(angles.shape)}')
    if n_rays <= 0:
        raise ValueError(f'n_rays must be positive, got {n_rays}')

    angles = angles.to(device=device, dtype=torch.float32)
    n_views = int(angles.numel())
    rows, cols = int(geo.n_detector[0]), int(geo.n_detector[1])
    d_row, d_col = float(geo.d_detector[0]), float(geo.d_detector[1])

    angle_idx = torch.randint(0, n_views, (n_rays,), device=device)
    row_idx = torch.randint(0, rows, (n_rays,), device=device)
    col_idx = torch.randint(0, cols, (n_rays,), device=device)
    sampled_angles = angles[angle_idx]

    cos_a = torch.cos(sampled_angles)
    sin_a = torch.sin(sampled_angles)

    origins = torch.stack(
        (-geo.DSO * sin_a, geo.DSO * cos_a, torch.zeros_like(cos_a)),
        dim=-1,
    )

    center_radius = float(geo.DSO - geo.DSD)
    detector_centers = torch.stack(
        (-center_radius * sin_a, center_radius * cos_a, torch.zeros_like(cos_a)),
        dim=-1,
    )
    detector_u = torch.stack((cos_a, sin_a, torch.zeros_like(cos_a)), dim=-1)
    detector_v = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32).expand(n_rays, 3)

    row_offsets = ((rows - 1.0) * 0.5 - row_idx.to(torch.float32)) * d_row
    col_offsets = (col_idx.to(torch.float32) - (cols - 1.0) * 0.5) * d_col

    detector_points = detector_centers + col_offsets[:, None] * detector_u + row_offsets[:, None] * detector_v
    directions = F.normalize(detector_points - origins, dim=-1)

    flat_indices = angle_idx * (rows * cols) + row_idx * cols + col_idx
    origins = _to_tigre_world(origins)
    directions = F.normalize(_to_tigre_world(directions), dim=-1)
    return origins.to(torch.float32), directions.to(torch.float32), flat_indices.to(torch.long)


def sample_points_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    n_samples: int,
    near: float,
    far: float,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points along each ray using stratified bins.

    Args:
        ray_origins: Ray origins with shape ``[N, 3]``.
        ray_directions: Ray directions with shape ``[N, 3]``.
        n_samples: Number of samples per ray.
        near: Near bound in millimeters.
        far: Far bound in millimeters.
        perturb: Whether to randomly jitter inside each sampling bin.

    Returns:
        Tuple ``(points, step_sizes)``:
            - ``points`` has shape ``[N, n_samples, 3]``.
            - ``step_sizes`` has shape ``[N, n_samples]``.
    """
    if ray_origins.ndim != 2 or ray_origins.shape[-1] != 3:
        raise ValueError(f'ray_origins must be [N,3], got shape={tuple(ray_origins.shape)}')
    if ray_directions.ndim != 2 or ray_directions.shape[-1] != 3:
        raise ValueError(f'ray_directions must be [N,3], got shape={tuple(ray_directions.shape)}')
    if ray_origins.shape[0] != ray_directions.shape[0]:
        raise ValueError('ray_origins and ray_directions must have same number of rays')
    if n_samples <= 0:
        raise ValueError(f'n_samples must be positive, got {n_samples}')
    if far <= near:
        raise ValueError(f'far must be greater than near, got near={near}, far={far}')

    n_rays = int(ray_origins.shape[0])
    device = ray_origins.device
    dtype = ray_origins.dtype

    bin_edges = torch.linspace(float(near), float(far), n_samples + 1, device=device, dtype=dtype)
    lower = bin_edges[:-1]
    upper = bin_edges[1:]
    step_widths = (upper - lower).unsqueeze(0).expand(n_rays, n_samples)

    if perturb:
        noise = torch.rand((n_rays, n_samples), device=device, dtype=dtype)
        t_vals = lower.unsqueeze(0) + noise * (upper - lower).unsqueeze(0)
    else:
        t_vals = ((lower + upper) * 0.5).unsqueeze(0).expand(n_rays, n_samples)

    points = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[..., None]
    return points.to(dtype=torch.float32), step_widths.to(dtype=torch.float32)


def normalize_coords(points: torch.Tensor, volume_size: list[float]) -> torch.Tensor:
    """Normalize world coordinates (mm) into ``[0, 1]^3`` volume coordinates.

    Args:
        points: Input coordinates with shape ``[..., 3]`` and origin at volume center.
        volume_size: Physical volume size ``[sx, sy, sz]`` in millimeters.

    Returns:
        Normalized coordinates with the same leading shape as ``points``.
    """
    if len(volume_size) != 3:
        raise ValueError(f'volume_size must have length 3, got {volume_size}')

    size = torch.as_tensor(volume_size, dtype=points.dtype, device=points.device)
    if torch.any(size <= 0):
        raise ValueError(f'volume_size values must be positive, got {volume_size}')

    return points / size + 0.5


def compute_near_far_naf(geo: CBCTGeometry, tolerance: float = 0.005) -> tuple[float, float]:
    """Compute NAF near/far in meters following official geometric bounds."""
    if tolerance < 0:
        raise ValueError(f'tolerance must be non-negative, got {tolerance}')
    dso = float(geo.DSO) / 1000.0
    s_voxel = np.asarray(geo.n_voxel, dtype=np.float32) * (np.asarray(geo.d_voxel, dtype=np.float32) / 1000.0)
    off_origin = np.zeros(3, dtype=np.float32)
    half_x = float(s_voxel[0]) * 0.5
    half_y = float(s_voxel[1]) * 0.5
    dist1 = np.linalg.norm([off_origin[0] - half_x, off_origin[1] - half_y])
    dist2 = np.linalg.norm([off_origin[0] - half_x, off_origin[1] + half_y])
    dist3 = np.linalg.norm([off_origin[0] + half_x, off_origin[1] - half_y])
    dist4 = np.linalg.norm([off_origin[0] + half_x, off_origin[1] + half_y])
    dist_max = float(np.max([dist1, dist2, dist3, dist4]))
    near = max(0.0, dso - dist_max - float(tolerance))
    far = min(dso * 2.0, dso + dist_max + float(tolerance))
    return near, far
