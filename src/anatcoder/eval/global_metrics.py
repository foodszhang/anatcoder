"""Global image-quality metrics for reconstructed volumes."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR between prediction and ground truth."""
    pred_f = np.asarray(pred, dtype=np.float64)
    gt_f = np.asarray(gt, dtype=np.float64)
    if pred_f.shape != gt_f.shape:
        raise ValueError(f'Shape mismatch: pred={pred_f.shape}, gt={gt_f.shape}')

    mse = float(np.mean((pred_f - gt_f) ** 2))
    if mse <= 1e-12:
        return float('inf')
    return float(20.0 * np.log10(float(data_range)) - 10.0 * np.log10(mse))


def _ssim_win_size(shape: tuple[int, ...]) -> int:
    """Pick a valid odd SSIM window size for 3D volumes."""
    min_dim = min(shape)
    win_size = 7 if min_dim >= 7 else min_dim
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(f'Volume is too small for SSIM computation: shape={shape}')
    return int(win_size)


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Compute 3D SSIM using ``skimage.metrics.structural_similarity``."""
    pred_f = np.asarray(pred, dtype=np.float64)
    gt_f = np.asarray(gt, dtype=np.float64)
    if pred_f.shape != gt_f.shape:
        raise ValueError(f'Shape mismatch: pred={pred_f.shape}, gt={gt_f.shape}')
    if pred_f.ndim != 3:
        raise ValueError(f'3D SSIM expects 3D input, got ndim={pred_f.ndim}')

    return float(
        structural_similarity(
            pred_f,
            gt_f,
            data_range=float(data_range),
            channel_axis=None,
            win_size=_ssim_win_size(pred_f.shape),
        )
    )


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mean absolute error."""
    pred_f = np.asarray(pred, dtype=np.float64)
    gt_f = np.asarray(gt, dtype=np.float64)
    if pred_f.shape != gt_f.shape:
        raise ValueError(f'Shape mismatch: pred={pred_f.shape}, gt={gt_f.shape}')
    return float(np.mean(np.abs(pred_f - gt_f)))


def evaluate_reconstruction(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> dict:
    """Compute all global reconstruction metrics in one call."""
    return {
        'psnr': compute_psnr(pred=pred, gt=gt, data_range=data_range),
        'ssim': compute_ssim(pred=pred, gt=gt, data_range=data_range),
        'mae': compute_mae(pred=pred, gt=gt),
    }


def psnr(pred_vol: np.ndarray, gt_vol: np.ndarray, data_range: float = 1.0) -> float:
    """Backward-compatible PSNR wrapper."""
    return compute_psnr(pred=pred_vol, gt=gt_vol, data_range=data_range)


def ssim(pred_vol: np.ndarray, gt_vol: np.ndarray, data_range: float = 1.0) -> float:
    """Backward-compatible SSIM wrapper."""
    return compute_ssim(pred=pred_vol, gt=gt_vol, data_range=data_range)


def mae(pred_vol: np.ndarray, gt_vol: np.ndarray) -> float:
    """Backward-compatible MAE wrapper."""
    return compute_mae(pred=pred_vol, gt=gt_vol)
