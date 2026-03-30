"""Vanilla INR baseline without anatomy conditioning."""

from typing import Any

from torch import Tensor, nn


class VanillaINRNetwork(nn.Module):
    """Hash-encoding plus MLP baseline with a single attenuation head."""

    def __init__(self, model_cfg: Any) -> None:
        """Initialize baseline encoding and MLP components."""
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, coords: Tensor) -> Tensor:
        """Predict attenuation coefficients at queried coordinates."""
        raise NotImplementedError("TODO: implement vanilla INR forward pass")
