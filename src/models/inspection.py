"""Helpers for lightweight model interpretability artifacts."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def extract_feature_importances(model, feature_names: Sequence[str]) -> list[dict[str, float | str]]:
    """Return sorted feature-importance pairs for supported sklearn estimators."""
    estimator = model.named_steps["classifier"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        raw_importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        raw_importances = np.abs(coefficients).mean(axis=0)
    else:
        raise ValueError("Model does not expose feature importance scores")

    feature_importances = [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in zip(feature_names, raw_importances, strict=True)
    ]
    return sorted(feature_importances, key=lambda item: item["importance"], reverse=True)
