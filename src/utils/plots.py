"""Plotting helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_confusion_matrix_heatmap(
    matrix: list[list[int]],
    output_path: str | Path,
    title: str,
) -> None:
    """Save a small confusion-matrix heatmap."""
    figure, axis = plt.subplots(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
