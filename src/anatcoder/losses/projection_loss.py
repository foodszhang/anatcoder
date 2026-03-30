"""Projection-domain reconstruction loss definitions."""

from torch import Tensor


def projection_mse_loss(pred_intensity: Tensor, gt_intensity: Tensor) -> Tensor:
    """Compute MSE between predicted and measured projection intensities."""
    raise NotImplementedError("TODO: implement projection MSE loss")
