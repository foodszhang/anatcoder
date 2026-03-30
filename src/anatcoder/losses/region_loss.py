"""Region-wise regularization losses for attenuation smoothness."""

from torch import Tensor


def region_variance_loss(mu: Tensor, class_probs: Tensor) -> Tensor:
    """Penalize within-region attenuation variance based on predicted anatomy."""
    raise NotImplementedError("TODO: implement region variance loss")
