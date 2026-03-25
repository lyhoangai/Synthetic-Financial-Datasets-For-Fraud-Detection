"""Training entry point for PaySim fraud baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TARGET_COLUMN
from src.data.load_data import load_paysim_csv
from src.data.split import split_dataset
from src.features.build_features import build_feature_matrix
from src.models.evaluate import compute_classification_metrics, select_best_threshold
from src.models.inspection import extract_feature_importances
from src.models.predict import predict_with_threshold
from src.rules.baseline_rules import apply_rule_baseline
from src.utils.io import write_json
from src.utils.plots import save_confusion_matrix_heatmap


MODELS = {
    "logistic_regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "random_forest": RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def maybe_sample_dataset(dataframe: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    """Optionally take a stratified sample to keep experiments lightweight."""
    if sample_size is None or sample_size >= len(dataframe):
        return dataframe

    if sample_size < dataframe[TARGET_COLUMN].nunique() * 2:
        raise ValueError("sample_size is too small for stratified sampling")

    sample_frame, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        dataframe,
        train_size=sample_size,
        stratify=dataframe[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return sample_frame.reset_index(drop=True)


def _align_feature_columns(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    return candidate.reindex(columns=reference.columns, fill_value=0)


def _resolve_figures_dir(output_root: Path) -> Path:
    """Place figures alongside the metrics directory under reports/."""
    if output_root.name == "metrics":
        return output_root.parent / "figures"
    if output_root.parent.name == "metrics":
        return output_root.parent.parent / "figures"
    return output_root.parent / "figures"


def _evaluate_rule_baseline(dataframe: pd.DataFrame) -> dict[str, float | int | list[list[int]]]:
    predictions = apply_rule_baseline(dataframe)
    return compute_classification_metrics(
        y_true=dataframe[TARGET_COLUMN],
        y_pred=predictions,
        y_score=predictions,
    )


def train_and_evaluate(
    input_path: str | Path,
    sample_size: int | None = None,
    output_dir: str | Path = "reports/metrics",
) -> dict[str, object]:
    """Run rule and ML baselines and export metrics."""
    raw_frame = load_paysim_csv(input_path)
    working_frame = maybe_sample_dataset(raw_frame, sample_size=sample_size)
    train_frame, validation_frame, test_frame = split_dataset(working_frame)

    train_features, train_target = build_feature_matrix(train_frame)
    validation_features, validation_target = build_feature_matrix(validation_frame)
    test_features, test_target = build_feature_matrix(test_frame)

    validation_features = _align_feature_columns(train_features, validation_features)
    test_features = _align_feature_columns(train_features, test_features)

    results: dict[str, object] = {
        "dataset": {
            "input_path": str(input_path),
            "rows_loaded": int(len(raw_frame)),
            "rows_used": int(len(working_frame)),
            "sample_size": sample_size,
        },
        "rule_baseline": {
            "validation": _evaluate_rule_baseline(validation_frame),
            "test": _evaluate_rule_baseline(test_frame),
        },
        "models": {},
    }
    threshold_artifacts: dict[str, object] = {}
    feature_importance_artifacts: dict[str, object] = {}

    for model_name, model in MODELS.items():
        model.fit(train_features, train_target)
        validation_scores = model.predict_proba(validation_features)[:, 1]
        threshold_selection = select_best_threshold(validation_target, validation_scores)
        selected_threshold = float(threshold_selection["best_threshold"])
        validation_predictions, _ = predict_with_threshold(
            model,
            validation_features,
            threshold=selected_threshold,
        )
        test_predictions, test_scores = predict_with_threshold(
            model,
            test_features,
            threshold=selected_threshold,
        )
        feature_importances = extract_feature_importances(model, train_features.columns.tolist())

        results["models"][model_name] = {
            "selected_threshold": threshold_selection,
            "validation": compute_classification_metrics(
                validation_target,
                validation_predictions,
                validation_scores,
            ),
            "test": compute_classification_metrics(
                test_target,
                test_predictions,
                test_scores,
            ),
            "feature_importances": feature_importances,
        }
        threshold_artifacts[model_name] = threshold_selection
        feature_importance_artifacts[model_name] = feature_importances

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(results, output_root / "model_comparison.json")
    write_json(threshold_artifacts, output_root / "threshold_selection.json")
    write_json(feature_importance_artifacts, output_root / "feature_importances.json")

    figures_root = _resolve_figures_dir(output_root)
    save_confusion_matrix_heatmap(
        results["rule_baseline"]["test"]["confusion_matrix"],
        figures_root / "rule_baseline_confusion_matrix.png",
        "Rule Baseline Confusion Matrix",
    )
    for model_name, model_results in results["models"].items():
        save_confusion_matrix_heatmap(
            model_results["test"]["confusion_matrix"],
            figures_root / f"{model_name}_confusion_matrix.png",
            f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        )

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the PaySim CSV file")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for quick local experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/metrics",
        help="Directory where JSON metrics will be written",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training CLI."""
    args = parse_args()
    train_and_evaluate(
        input_path=args.input,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
