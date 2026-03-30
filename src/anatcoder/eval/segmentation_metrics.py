"""Segmentation quality metrics for emergent anatomy predictions."""

import numpy as np


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int | None = None) -> float:
    """Compute Dice score for one class or all foreground classes."""
    raise NotImplementedError("TODO: implement Dice metric")


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int | None = None) -> float:
    """Compute IoU score for one class or all foreground classes."""
    raise NotImplementedError("TODO: implement IoU metric")
