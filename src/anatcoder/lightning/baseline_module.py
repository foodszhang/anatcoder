"""Unified Lightning interface for external baseline wrappers."""

from typing import Any

from .base_module import BaseReconModule


class BaselineModule(BaseReconModule):
    """Module adapter for methods wrapped from external repositories."""

    def __init__(self, cfg: Any) -> None:
        """Initialize selected external baseline adapter."""
        super().__init__(cfg)

    def forward_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Forward batch by delegating to the selected external method."""
        raise NotImplementedError("TODO: implement external baseline forward_step")

    def compute_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Compute standardized losses for wrapped baseline methods."""
        raise NotImplementedError("TODO: implement external baseline loss computation")
