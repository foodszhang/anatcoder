"""Wrapper interface for integrating the NAF baseline."""

from pathlib import Path
from typing import Any


class NAFWrapper:
    """Adapter class for training/inference with an external NAF implementation."""

    def __init__(self, config: Any) -> None:
        """Store configuration for the wrapped NAF model."""
        self.config = config

    def fit(self, train_data: Any) -> None:
        """Train or fine-tune the wrapped NAF model."""
        raise NotImplementedError("TODO: implement NAF training wrapper")

    def predict(self, input_data: Any, output_dir: str | Path) -> Any:
        """Run NAF inference and persist artifacts."""
        raise NotImplementedError("TODO: implement NAF inference wrapper")
