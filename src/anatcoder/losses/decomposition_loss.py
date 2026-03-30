"""Losses for anatomy-decomposed volume rendering supervision."""

from torch import Tensor


def decomposition_kl_loss(pred_decomposition: Tensor, atlas_decomposition: Tensor) -> Tensor:
    """Compute KL divergence between predicted and atlas decomposition terms."""
    raise NotImplementedError("TODO: implement decomposition KL loss")
