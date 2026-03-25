"""Evaluation utilities for fraud models."""

from __future__ import annotations

from math import isclose
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


def select_best_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    """Pick the threshold with the best validation F1 score."""
    candidate_thresholds = list(thresholds or [round(value / 100, 2) for value in range(10, 95, 5)])
    true_values = list(y_true)
    score_values = list(y_score)

    candidates: list[dict[str, float]] = []
    best_candidate: dict[str, float] | None = None
    for threshold in candidate_thresholds:
        predictions = [1 if score >= threshold else 0 for score in score_values]
        metrics = compute_classification_metrics(true_values, predictions, score_values)
        candidate = {
            "threshold": float(threshold),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
        }
        candidates.append(candidate)

        if best_candidate is None:
            best_candidate = candidate
            continue

        if candidate["f1"] > best_candidate["f1"]:
            best_candidate = candidate
            continue

        if isclose(candidate["f1"], best_candidate["f1"]):
            if candidate["precision"] > best_candidate["precision"]:
                best_candidate = candidate
            elif isclose(candidate["precision"], best_candidate["precision"]) and candidate["threshold"] < best_candidate["threshold"]:
                best_candidate = candidate

    assert best_candidate is not None
    return {
        "best_threshold": best_candidate["threshold"],
        "best_metric": "f1",
        "best_value": best_candidate["f1"],
        "candidates": candidates,
    }
