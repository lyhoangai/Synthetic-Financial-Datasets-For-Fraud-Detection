from src.models.evaluate import compute_classification_metrics, select_best_threshold


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


def test_select_best_threshold_uses_validation_f1():
    threshold_selection = select_best_threshold(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.6, 0.7, 0.95],
        thresholds=[0.5, 0.7, 0.9],
    )

    assert threshold_selection["best_threshold"] == 0.7
    assert threshold_selection["best_metric"] == "f1"
    assert threshold_selection["best_value"] == 1.0
    assert threshold_selection["candidates"][0]["threshold"] == 0.5
