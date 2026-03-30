"""Ray generation and sampling utility functions."""

from typing import Any

from torch import Tensor


def generate_rays(
    geo_params: dict[str, Any],
    angles: Tensor,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Generate ray origins and directions for each detector pixel."""
    raise NotImplementedError("TODO: implement CBCT ray generation")


def sample_points_along_ray(
    rays_o: Tensor,
    rays_d: Tensor,
    near: float,
    far: float,
    n_samples: int,
) -> tuple[Tensor, Tensor]:
    """Sample 3D points and step sizes along rays for volume rendering."""
    raise NotImplementedError("TODO: implement ray point sampling")
