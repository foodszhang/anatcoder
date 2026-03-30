"""Wrapper interface for integrating the Spener baseline."""

from pathlib import Path
from typing import Any


class SpenerWrapper:
    """Adapter class for training/inference with an external Spener implementation."""

    def __init__(self, config: Any) -> None:
        """Store configuration for the wrapped Spener model."""
        self.config = config

    def fit(self, train_data: Any) -> None:
        """Train or fine-tune the wrapped Spener model."""
        raise NotImplementedError("TODO: implement Spener training wrapper")

    def predict(self, input_data: Any, output_dir: str | Path) -> Any:
        """Run Spener inference and persist artifacts."""
        raise NotImplementedError("TODO: implement Spener inference wrapper")
