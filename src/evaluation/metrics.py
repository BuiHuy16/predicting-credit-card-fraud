"""Evaluation metrics calculation for binary fraud classification."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Compute primary (ROC-AUC), secondary (PR-AUC), and threshold-based metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_score: Predicted probabilities for positive class (is_fraud=1).
        threshold: Decision threshold for discrete classification metrics.

    Returns:
        Dictionary of computed metric values.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1]; received {threshold}")
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true_arr)}) vs y_score ({len(y_score_arr)})"
        )
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("y_true must contain both positive and negative classes.")

    roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_score_arr))

    y_pred = (y_score_arr >= threshold).astype(int)
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred).ravel()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
