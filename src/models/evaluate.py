"""Evaluation utilities for fraud models."""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float],
) -> dict[str, float | int | list[list[int]]]:
    """Compute metrics that matter for imbalanced fraud detection."""
    true_values = list(y_true)
    predicted_values = list(y_pred)
    score_values = list(y_score)
    tn, fp, fn, tp = confusion_matrix(true_values, predicted_values, labels=[0, 1]).ravel()

    return {
        "precision": float(precision_score(true_values, predicted_values, zero_division=0)),
        "recall": float(recall_score(true_values, predicted_values, zero_division=0)),
        "f1": float(f1_score(true_values, predicted_values, zero_division=0)),
        "pr_auc": float(average_precision_score(true_values, score_values)),
        "support_positive": int(sum(true_values)),
        "support_negative": int(len(true_values) - sum(true_values)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
