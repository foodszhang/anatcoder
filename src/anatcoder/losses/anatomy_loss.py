"""Anatomy consistency loss definitions."""

from torch import Tensor


def anatomy_kl_loss(pred_probs: Tensor, atlas_probs: Tensor) -> Tensor:
    """Compute KL divergence between predicted and atlas anatomy distributions."""
    raise NotImplementedError("TODO: implement anatomy KL loss")
