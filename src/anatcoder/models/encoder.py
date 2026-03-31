"""Coordinate encoders for neural implicit CT reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HashGridEncoder(nn.Module):
    """Multi-resolution hash-grid encoder with tiny-cuda-nn fallback behavior."""

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.4472,
    ) -> None:
        """Initialize hash-grid encoder and backend implementation."""
        super().__init__()
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.base_resolution = int(base_resolution)
        self.per_level_scale = float(per_level_scale)
        self._output_dim = self.n_levels * self.n_features_per_level

        self._backend = 'torch'
        self._encoding: nn.Module | None = None
        try:
            import tinycudann as tcnn

            self._encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': self.n_levels,
                    'n_features_per_level': self.n_features_per_level,
                    'log2_hashmap_size': self.log2_hashmap_size,
                    'base_resolution': self.base_resolution,
                    'per_level_scale': self.per_level_scale,
                },
            )
            self._backend = 'tcnn'
        except Exception:
            self._encoding = None
            self._backend = 'torch'

    @property
    def output_dim(self) -> int:
        """Return encoder output feature dimension."""
        return self._output_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode normalized coordinates into feature vectors.

        Args:
            coords: Input tensor with shape ``[..., 3]`` in normalized range ``[0, 1]``.

        Returns:
            Encoded features with shape ``[..., output_dim]``.
        """
        if coords.shape[-1] != 3:
            raise ValueError(f'coords last dim must be 3, got shape={tuple(coords.shape)}')

        original_shape = coords.shape[:-1]
        flat = coords.reshape(-1, 3).to(torch.float32)
        flat = torch.clamp(flat, 0.0, 1.0)

        if self._backend == 'tcnn' and self._encoding is not None:
            encoded = self._encoding(flat)
            return encoded.reshape(*original_shape, self.output_dim)

        # Fallback: deterministic multi-frequency sinusoidal features shaped to output_dim.
        freqs = torch.arange(
            1, self.n_levels + 1, device=flat.device, dtype=flat.dtype
        ) * self.per_level_scale
        phase = flat[:, :, None] * freqs[None, None, :] * (2.0 * math.pi)
        sin_feat = torch.sin(phase)
        cos_feat = torch.cos(phase)
        mixed = torch.cat([sin_feat, cos_feat], dim=1).permute(0, 2, 1).reshape(flat.shape[0], -1)
        if mixed.shape[1] < self.output_dim:
            pad = self.output_dim - mixed.shape[1]
            mixed = torch.cat([mixed, torch.zeros(flat.shape[0], pad, device=flat.device, dtype=flat.dtype)], dim=1)
        encoded = mixed[:, : self.output_dim]
        return encoded.reshape(*original_shape, self.output_dim)


class PositionalEncoder(nn.Module):
    """NeRF-style sinusoidal positional encoder."""

    def __init__(self, n_freqs: int = 10) -> None:
        """Initialize positional encoder frequency bands."""
        super().__init__()
        self.n_freqs = int(n_freqs)
        if self.n_freqs <= 0:
            raise ValueError(f'n_freqs must be positive, got {self.n_freqs}')
        self.register_buffer(
            '_freq_bands',
            2.0 ** torch.arange(self.n_freqs, dtype=torch.float32),
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        """Return positional encoding output dimension."""
        return 3 + 6 * self.n_freqs

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with sinusoidal features.

        Args:
            coords: Input tensor with shape ``[..., 3]``.

        Returns:
            Encoded tensor with shape ``[..., output_dim]``.
        """
        if coords.shape[-1] != 3:
            raise ValueError(f'coords last dim must be 3, got shape={tuple(coords.shape)}')
        original_shape = coords.shape[:-1]
        flat = coords.reshape(-1, 3).to(torch.float32)
        flat = torch.clamp(flat, 0.0, 1.0)

        phases = flat[:, None, :] * self._freq_bands[None, :, None] * math.pi
        sin_features = torch.sin(phases).reshape(flat.shape[0], -1)
        cos_features = torch.cos(phases).reshape(flat.shape[0], -1)
        encoded = torch.cat([flat, sin_features, cos_features], dim=-1)
        return encoded.reshape(*original_shape, self.output_dim)

