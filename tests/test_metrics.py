from src.models.evaluate import compute_classification_metrics


def test_compute_classification_metrics_returns_core_scores():
    metrics = compute_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_score=[0.1, 0.8, 0.9, 0.95],
    )

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "pr_auc" in metrics
    assert metrics["support_positive"] == 2
