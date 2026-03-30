"""Base Lightning module defining shared optimization and logging behavior."""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L


class BaseReconModule(L.LightningModule, ABC):
    """Shared trainer hooks for reconstruction methods."""

    def __init__(self, cfg: Any) -> None:
        """Store configuration used by derived modules."""
        super().__init__()
        self.cfg = cfg

    def configure_optimizers(self):
        """Configure optimizer and optional learning-rate scheduler."""
        raise NotImplementedError("TODO: implement optimizer setup")

    @abstractmethod
    def forward_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run method-specific forward pass for one training batch."""

    @abstractmethod
    def compute_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Compute method-specific losses from outputs and targets."""
