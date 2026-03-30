"""Tests for renderer module scaffolding."""

from pathlib import Path


def test_renderer_module_exists() -> None:
    """Ensure renderer scaffold file is present."""
    assert Path("src/anatcoder/models/components/renderers.py").exists()
