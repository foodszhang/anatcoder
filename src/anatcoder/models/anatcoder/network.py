"""Main AnatCoder network that predicts attenuation and anatomy probabilities."""

from typing import Any

from torch import Tensor, nn


class AnatCoderNetwork(nn.Module):
    """Anatomy-conditioned implicit representation with dual prediction heads."""

    def __init__(self, model_cfg: Any) -> None:
        """Initialize encoder, conditioning, backbone, and output heads."""
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Predict attenuation coefficients and anatomy probabilities."""
        raise NotImplementedError("TODO: implement AnatCoder forward pipeline")
