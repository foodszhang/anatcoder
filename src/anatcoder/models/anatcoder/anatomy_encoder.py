"""Anatomical probability embedding module."""

from torch import Tensor, nn


class AnatomyEncoder(nn.Module):
    """Encode atlas class probabilities into compact conditioning vectors."""

    def __init__(self, n_classes: int = 104, embed_dim: int = 64) -> None:
        """Initialize anatomy embedding projection layers."""
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

    def forward(self, atlas_probs: Tensor) -> Tensor:
        """Map atlas probabilities [N, C] to embeddings [N, embed_dim]."""
        raise NotImplementedError("TODO: implement anatomy embedding network")
