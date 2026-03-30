"""Tests for projection module scaffolding."""

from pathlib import Path


def test_projection_module_exists() -> None:
    """Ensure projection scaffold file is present."""
    assert Path("src/anatcoder/data/projection.py").exists()
