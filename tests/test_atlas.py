"""Tests for atlas module scaffolding."""

from pathlib import Path


def test_atlas_module_exists() -> None:
    """Ensure atlas scaffold file is present."""
    assert Path("src/anatcoder/data/atlas.py").exists()
