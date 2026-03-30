"""Global image-quality metrics for reconstructed volumes."""

import numpy as np


def psnr(pred_vol: np.ndarray, gt_vol: np.ndarray, data_range: float = 1.0) -> float:
    """Compute peak signal-to-noise ratio for full 3D volumes."""
    raise NotImplementedError("TODO: implement PSNR metric")


def ssim(pred_vol: np.ndarray, gt_vol: np.ndarray, data_range: float = 1.0) -> float:
    """Compute structural similarity for full 3D volumes."""
    raise NotImplementedError("TODO: implement SSIM metric")


def mae(pred_vol: np.ndarray, gt_vol: np.ndarray) -> float:
    """Compute mean absolute error for full 3D volumes."""
    raise NotImplementedError("TODO: implement MAE metric")
