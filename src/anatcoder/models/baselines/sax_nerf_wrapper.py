"""Wrapper interface for integrating the SAX-NeRF baseline."""

from pathlib import Path
from typing import Any


class SAXNeRFWrapper:
    """Adapter class for training/inference with an external SAX-NeRF implementation."""

    def __init__(self, config: Any) -> None:
        """Store configuration for the wrapped SAX-NeRF model."""
        self.config = config

    def fit(self, train_data: Any) -> None:
        """Train or fine-tune the wrapped SAX-NeRF model."""
        raise NotImplementedError("TODO: implement SAX-NeRF training wrapper")

    def predict(self, input_data: Any, output_dir: str | Path) -> Any:
        """Run SAX-NeRF inference and persist artifacts."""
        raise NotImplementedError("TODO: implement SAX-NeRF inference wrapper")
