"""Vanilla INR network mapping normalized coordinates to attenuation values."""

from __future__ import annotations

from typing import Sequence

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
        skips: Sequence[int] | None = None,
        last_activation: str = 'softplus',
        n_anatomy_classes: int = 0,
    ) -> None:
        """Initialize encoder and MLP backbone."""
        super().__init__()
        if n_hidden_layers <= 0:
            raise ValueError(f'n_hidden_layers must be positive, got {n_hidden_layers}')
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be positive, got {hidden_dim}')
        if n_anatomy_classes < 0:
            raise ValueError(f'n_anatomy_classes must be >=0, got {n_anatomy_classes}')

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

        self._encoder_dim = int(self.encoder.output_dim)
        self._n_anatomy_classes = int(n_anatomy_classes)
        self._input_dim = self._encoder_dim + self._n_anatomy_classes
        self._hidden_dim = int(hidden_dim)
        self._skips = set(int(i) for i in skips) if skips is not None else set()
        self.bound: float | None = None
        invalid_skips = [i for i in self._skips if i < 0 or i >= n_hidden_layers]
        if invalid_skips:
            raise ValueError(f'skip indices out of range: {invalid_skips}, n_hidden_layers={n_hidden_layers}')

        self._mlp_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if i == 0:
                in_dim = self._input_dim
            else:
                in_dim = self._hidden_dim
            if i in self._skips:
                in_dim += self._input_dim
            self._mlp_layers.append(nn.Linear(in_dim, self._hidden_dim))

        # Keep backward-compatible attribute name used elsewhere in training code.
        self.mlp = self._mlp_layers
        self.attenuation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        act = str(last_activation).lower()
        if act == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        elif act == 'softplus':
            self.out_activation = nn.Softplus()
        else:
            raise ValueError(f'Unsupported last_activation: {last_activation}')

    def forward(
        self,
        coords: torch.Tensor,
        anatomy_onehot: torch.Tensor | None = None,
        anatomy_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict attenuation coefficients for normalized coordinates.

        Args:
            coords: Tensor with shape ``[N, 3]`` in ``[0, 1]^3``.
            anatomy_onehot: Optional one-hot anatomy conditions with shape ``[N, C]``.
            anatomy_labels: Optional integer anatomy labels with shape ``[N]`` or ``[N,1]``.

        Returns:
            Predicted attenuation ``mu`` with shape ``[N, 1]``.
        """
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f'coords must be [N,3], got shape={tuple(coords.shape)}')
        encoded = self.encoder(coords)
        if self._n_anatomy_classes > 0:
            if anatomy_onehot is None and anatomy_labels is not None:
                labels = anatomy_labels
                if labels.ndim == 2 and labels.shape[1] == 1:
                    labels = labels.squeeze(-1)
                if labels.ndim != 1 or labels.shape[0] != coords.shape[0]:
                    raise ValueError(
                        'anatomy_labels must be [N] or [N,1] with N matching coords, '
                        f'got shape={tuple(anatomy_labels.shape)} and coords={tuple(coords.shape)}'
                    )
                labels = labels.to(dtype=torch.long, device=encoded.device).clamp(0, self._n_anatomy_classes - 1)
                anatomy_onehot = torch.nn.functional.one_hot(labels, num_classes=self._n_anatomy_classes).to(
                    dtype=encoded.dtype
                )
            if anatomy_onehot is None:
                anatomy_onehot = torch.zeros(
                    (coords.shape[0], self._n_anatomy_classes),
                    dtype=encoded.dtype,
                    device=encoded.device,
                )
            if anatomy_onehot.ndim != 2 or anatomy_onehot.shape[0] != coords.shape[0]:
                raise ValueError(
                    'anatomy_onehot must be [N,C] with N matching coords, '
                    f'got shape={tuple(anatomy_onehot.shape)} and coords={tuple(coords.shape)}'
                )
            if anatomy_onehot.shape[1] != self._n_anatomy_classes:
                raise ValueError(
                    f'anatomy_onehot classes mismatch: expected {self._n_anatomy_classes}, '
                    f'got {anatomy_onehot.shape[1]}'
                )
            if anatomy_onehot.dtype != encoded.dtype:
                anatomy_onehot = anatomy_onehot.to(dtype=encoded.dtype)
            if anatomy_onehot.device != encoded.device:
                anatomy_onehot = anatomy_onehot.to(device=encoded.device)
            encoded = torch.cat([encoded, anatomy_onehot], dim=-1)

        # tinycudann hash-grid can emit fp16 features on CUDA; align with MLP weights.
        target_dtype = self._mlp_layers[0].weight.dtype
        if encoded.dtype != target_dtype:
            encoded = encoded.to(dtype=target_dtype)
        h = encoded
        for i, layer in enumerate(self._mlp_layers):
            if i in self._skips:
                h = torch.cat([h, encoded], dim=-1)
            h = torch.relu(layer(h))
        features = h
        mu = self.out_activation(self.attenuation_head(features))
        return mu

    def query_density(
        self,
        coords: torch.Tensor,
        anatomy_onehot: torch.Tensor | None = None,
        anatomy_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias of :meth:`forward` for semantic clarity in rendering code."""
        return self.forward(coords, anatomy_onehot=anatomy_onehot, anatomy_labels=anatomy_labels)
