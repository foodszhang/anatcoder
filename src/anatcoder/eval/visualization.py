"""Visualization helpers for qualitative reconstruction analysis."""

from pathlib import Path

import numpy as np


def render_organ_heatmap(volume: np.ndarray, seg_mask: np.ndarray, output_path: str | Path) -> None:
    """Render organ heatmap overlays for qualitative comparison."""
    raise NotImplementedError("TODO: implement organ heatmap visualization")


def render_advr_decomposition(decomposition: np.ndarray, output_path: str | Path) -> None:
    """Render anatomy-decomposed attenuation components."""
    raise NotImplementedError("TODO: implement ADVR decomposition visualization")


def render_line_profile(pred_line: np.ndarray, gt_line: np.ndarray, output_path: str | Path) -> None:
    """Plot line-profile comparisons between prediction and reference."""
    raise NotImplementedError("TODO: implement line profile visualization")
