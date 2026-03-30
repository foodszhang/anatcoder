"""Configurable MLP backbone with optional skip connections."""

from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """A generic MLP used by both AnatCoder and baseline models."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        skip_connections: Sequence[int] | None = None,
    ) -> None:
        """Build a fully-connected network layout."""
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.skip_connections = list(skip_connections or [])

    def forward(self, x: Tensor) -> Tensor:
        """Run MLP forward pass."""
        raise NotImplementedError("TODO: implement configurable MLP")
