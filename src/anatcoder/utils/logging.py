"""Experiment logger factory utilities."""

from typing import Any


def build_experiment_logger(logger_cfg: Any) -> Any:
    """Build and return the configured experiment logger instance."""
    raise NotImplementedError("TODO: implement logger factory for WandB/TensorBoard")
