"""Lightning module for the Vanilla INR baseline."""

from typing import Any

from .base_module import BaseReconModule


class VanillaINRModule(BaseReconModule):
    """Training module for hash-MLP attenuation-only baseline."""

    def __init__(self, cfg: Any) -> None:
        """Initialize Vanilla INR model components."""
        super().__init__(cfg)

    def forward_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Forward one batch through the Vanilla INR network."""
        raise NotImplementedError("TODO: implement Vanilla INR forward_step")

    def compute_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Compute Vanilla INR training losses."""
        raise NotImplementedError("TODO: implement Vanilla INR loss computation")
