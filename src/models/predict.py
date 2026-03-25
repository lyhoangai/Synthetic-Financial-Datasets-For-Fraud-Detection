"""Prediction helpers for trained classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_with_threshold(model, features: pd.DataFrame, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and scores using a configurable threshold."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
