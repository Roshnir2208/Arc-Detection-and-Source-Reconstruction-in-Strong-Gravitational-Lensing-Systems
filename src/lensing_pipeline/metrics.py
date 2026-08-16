from __future__ import annotations

import math

import numpy as np


# Compare predicted and true masks at pixel level using TP, FP, FN, and TN.
def segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)

    tp = int(np.logical_and(pred, true).sum())
    fp = int(np.logical_and(pred, ~true).sum())
    fn = int(np.logical_and(~pred, true).sum())
    tn = int(np.logical_and(~pred, ~true).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# Compute peak signal-to-noise ratio for reconstruction quality.
def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


# Match reconstructed brightness to the reference using one least-squares scale factor.
def match_photometric_scale(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    denom = float(np.sum(est * est))
    if denom <= 0:
        return est
    scale = float(np.sum(ref * est)) / denom
    return np.clip(scale * est, 0.0, 1.0)


# Compute a lightweight global SSIM-style similarity score.
def ssim_simple(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(ref.mean())
    mu_y = float(est.mean())
    var_x = float(ref.var())
    var_y = float(est.var())
    cov = float(((ref - mu_x) * (est - mu_y)).mean())
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
