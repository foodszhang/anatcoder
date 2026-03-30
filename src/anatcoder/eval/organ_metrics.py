"""Organ-level evaluation metrics and grouped reporting utilities."""

from collections.abc import Mapping
from typing import Any

import numpy as np


class OrganEvaluator:
    """Evaluate per-organ reconstruction quality and HU bias."""

    def __init__(self, seg_mask: np.ndarray, organ_groups: Mapping[str, list[str]]) -> None:
        """Store segmentation mask and organ grouping definitions."""
        self.seg_mask = seg_mask
        self.organ_groups = organ_groups

    def evaluate(self, pred_vol: np.ndarray, gt_vol: np.ndarray) -> dict[str, Any]:
        """Return organ-wise metrics as a dictionary."""
        raise NotImplementedError("TODO: implement organ-level evaluation")
