"""Hash-grid encoding wrapper (tiny-cuda-nn backend)."""

from typing import Any

from torch import Tensor, nn


class HashGridEncoding(nn.Module):
    """Encode 3D coordinates into high-dimensional multi-resolution features."""

    def __init__(
        self,
        n_levels: int,
        n_features: int,
        log2_hashmap_size: int,
        base_resolution: int = 16,
        per_level_scale: float = 1.447,
        **kwargs: Any,
    ) -> None:
        """Configure hash-grid encoding hyperparameters."""
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.kwargs = kwargs

    def forward(self, coords: Tensor) -> Tensor:
        """Encode coordinates of shape [N, 3] into hash features."""
        raise NotImplementedError("TODO: integrate tiny-cuda-nn hash encoding")
