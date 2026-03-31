"""Vanilla INR network mapping normalized coordinates to attenuation values."""

from __future__ import annotations

import torch
import torch.nn as nn

from anatcoder.models.encoder import HashGridEncoder, PositionalEncoder


class VanillaINR(nn.Module):
    """Hash-grid/positional encoder + MLP baseline for attenuation regression."""

    def __init__(
        self,
        encoder_type: str = 'hashgrid',
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.4472,
        n_hidden_layers: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize encoder and MLP backbone."""
        super().__init__()
        if n_hidden_layers <= 0:
            raise ValueError(f'n_hidden_layers must be positive, got {n_hidden_layers}')
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be positive, got {hidden_dim}')

        encoder_name = encoder_type.lower()
        if encoder_name in {'hashgrid', 'hash_grid'}:
            self.encoder = HashGridEncoder(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                per_level_scale=per_level_scale,
            )
            self.encoder_type = 'hashgrid'
        elif encoder_name in {'positional', 'pe'}:
            self.encoder = PositionalEncoder(n_freqs=10)
            self.encoder_type = 'positional'
        else:
            raise ValueError(f'Unsupported encoder_type: {encoder_type}')

        layers: list[nn.Module] = []
        in_dim = self.encoder.output_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.out_activation = nn.Softplus()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict attenuation coefficients for normalized coordinates.

        Args:
            coords: Tensor with shape ``[N, 3]`` in ``[0, 1]^3``.

        Returns:
            Predicted attenuation ``mu`` with shape ``[N, 1]``.
        """
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f'coords must be [N,3], got shape={tuple(coords.shape)}')
        encoded = self.encoder(coords)
        mu = self.out_activation(self.mlp(encoded))
        return mu

    def query_density(self, coords: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward` for semantic clarity in rendering code."""
        return self.forward(coords)

