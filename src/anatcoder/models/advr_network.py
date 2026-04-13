"""ADVR model: shared encoder/backbone with anatomy-specific attenuation heads."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from anatcoder.models.encoder import HashGridEncoder, PositionalEncoder


class ADVRNetwork(nn.Module):
    """Shared-feature INR with anatomy-routed attenuation heads."""

    def __init__(
        self,
        encoder_type: str = 'hashgrid',
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.4472,
        n_hidden_layers: int = 2,
        hidden_dim: int = 256,
        head_hidden_dim: int = 64,
        last_activation: str = 'sigmoid',
        n_anatomy_classes: int = 10,
        skips: Sequence[int] | None = None,
    ) -> None:
        """Initialize shared encoder/backbone and class-specific heads."""
        super().__init__()
        _ = skips
        if n_hidden_layers <= 0:
            raise ValueError(f'n_hidden_layers must be positive, got {n_hidden_layers}')
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be positive, got {hidden_dim}')
        if head_hidden_dim <= 0:
            raise ValueError(f'head_hidden_dim must be positive, got {head_hidden_dim}')
        if n_anatomy_classes <= 0:
            raise ValueError(f'n_anatomy_classes must be positive for ADVR, got {n_anatomy_classes}')

        encoder_name = encoder_type.lower()
        if encoder_name in {'hashgrid', 'hash_grid'}:
            self.encoder = HashGridEncoder(
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                per_level_scale=per_level_scale,
            )
        elif encoder_name in {'positional', 'pe'}:
            self.encoder = PositionalEncoder(n_freqs=10)
        else:
            raise ValueError(f'Unsupported encoder_type: {encoder_type}')

        self.n_anatomy_classes = int(n_anatomy_classes)
        self.hidden_dim = int(hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.n_hidden_layers = int(n_hidden_layers)

        shared_layers: list[nn.Module] = []
        in_dim = int(self.encoder.output_dim)
        for _layer_idx in range(self.n_hidden_layers):
            shared_layers.append(nn.Linear(in_dim, self.hidden_dim))
            shared_layers.append(nn.ReLU(inplace=True))
            in_dim = self.hidden_dim
        self.shared_mlp = nn.Sequential(*shared_layers)

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.head_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.head_hidden_dim, 1),
                )
                for _ in range(self.n_anatomy_classes)
            ]
        )
        act = str(last_activation).lower()
        if act == 'sigmoid':
            self.out_activation: nn.Module = nn.Sigmoid()
        elif act == 'softplus':
            self.out_activation = nn.Softplus()
        else:
            raise ValueError(f'Unsupported last_activation: {last_activation}')
        self.bound: float | None = None

    def _normalize_labels(self, anatomy_labels: torch.Tensor, num_points: int) -> torch.Tensor:
        """Normalize label tensor to ``[N]`` long labels with valid class range."""
        labels = anatomy_labels
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(-1)
        if labels.ndim != 1 or labels.shape[0] != num_points:
            raise ValueError(
                f'anatomy_labels must be [N] or [N,1] matching coords, got {tuple(anatomy_labels.shape)}'
            )
        if labels.dtype != torch.long:
            labels = labels.to(torch.long)
        return labels.clamp(0, self.n_anatomy_classes - 1)

    def forward(
        self,
        coords: torch.Tensor,
        anatomy_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict attenuation for coordinates using anatomy-routed heads."""
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f'coords must be [N,3], got shape={tuple(coords.shape)}')

        encoded = self.encoder(coords)
        target_dtype = self.shared_mlp[0].weight.dtype
        if encoded.dtype != target_dtype:
            encoded = encoded.to(dtype=target_dtype)
        shared_feat = self.shared_mlp(encoded)

        if anatomy_labels is None:
            return self.out_activation(self.heads[0](shared_feat))

        labels = self._normalize_labels(anatomy_labels, num_points=coords.shape[0]).to(device=coords.device)
        output = torch.zeros((coords.shape[0], 1), dtype=shared_feat.dtype, device=shared_feat.device)
        for class_idx, head in enumerate(self.heads):
            mask = labels == class_idx
            if torch.any(mask):
                output[mask] = head(shared_feat[mask])
        return self.out_activation(output)

    def query_density(
        self,
        coords: torch.Tensor,
        anatomy_labels: torch.Tensor | None = None,
        anatomy_onehot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward-compatible density query supporting label or one-hot conditioning."""
        labels = anatomy_labels
        if labels is None and anatomy_onehot is not None:
            if anatomy_onehot.ndim != 2 or anatomy_onehot.shape[0] != coords.shape[0]:
                raise ValueError(
                    'anatomy_onehot must be [N,C] with N matching coords, '
                    f'got shape={tuple(anatomy_onehot.shape)}'
                )
            if anatomy_onehot.shape[1] != self.n_anatomy_classes:
                raise ValueError(
                    f'anatomy_onehot classes mismatch: expected {self.n_anatomy_classes}, '
                    f'got {anatomy_onehot.shape[1]}'
                )
            labels = torch.argmax(anatomy_onehot, dim=-1)
        return self.forward(coords, anatomy_labels=labels)

