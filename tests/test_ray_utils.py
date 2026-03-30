"""Tests for ray utility module scaffolding."""

from pathlib import Path


def test_ray_utils_module_exists() -> None:
    """Ensure ray utility scaffold file is present."""
    assert Path("src/anatcoder/models/components/ray_utils.py").exists()
