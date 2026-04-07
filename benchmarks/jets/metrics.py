"""Jet classification metrics: b-tag AUC, flavour rejection rates."""

from __future__ import annotations

import numpy as np


def roc_auc(y_true, y_score) -> float:
    """Simple AUC computation (trapezoidal rule, no sklearn dependency)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    # Prepend (0,0)
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    # np.trapezoid (new name in numpy 2.x) with a trapz fallback for 1.x
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return round(float(trap(tpr, fpr)), 6)


def rejection_at_efficiency(y_true, y_score, target_eff: float) -> float:
    """Background rejection (1/FPR) at a chosen signal efficiency."""
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    sig_scores = np.sort(y_score[y_true])[::-1]
    bg_scores = y_score[~y_true]
    if len(sig_scores) == 0:
        return 0.0
    idx = int(len(sig_scores) * target_eff)
    idx = min(idx, len(sig_scores) - 1)
    threshold = sig_scores[idx]
    fpr = float((bg_scores >= threshold).mean())
    if fpr == 0:
        return float("inf")
    return round(1.0 / fpr, 3)
