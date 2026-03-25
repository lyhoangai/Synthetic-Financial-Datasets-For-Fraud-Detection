from pathlib import Path

import pandas as pd

from src.models.train import train_and_evaluate


def _make_training_sample(rows: int = 60) -> pd.DataFrame:
    records = []
    for idx in range(rows):
        is_fraud = 1 if idx % 5 == 0 else 0
        tx_type = "TRANSFER" if idx % 2 == 0 else "CASH_OUT"
        old_balance = float(500 + idx)
        amount = old_balance if is_fraud else float(100 + idx)
        new_balance_orig = 0.0 if is_fraud else old_balance - amount
        records.append(
            {
                "step": idx + 1,
                "type": tx_type,
                "amount": amount,
                "nameOrig": f"C{idx:06d}",
                "oldbalanceOrg": old_balance,
                "newbalanceOrig": new_balance_orig,
                "nameDest": f"M{idx:06d}",
                "oldbalanceDest": float(200 + idx),
                "newbalanceDest": float(200 + idx + amount),
                "isFraud": is_fraud,
                "isFlaggedFraud": 1 if is_fraud and idx % 10 == 0 else 0,
            }
        )
    return pd.DataFrame(records)


def test_train_and_evaluate_writes_metrics_and_figures(tmp_path: Path):
    dataframe = _make_training_sample()
    csv_path = tmp_path / "raw" / "sample.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)

    output_dir = tmp_path / "reports" / "metrics" / "smoke"
    results = train_and_evaluate(csv_path, output_dir=output_dir)

    assert (output_dir / "model_comparison.json").exists()
    assert (tmp_path / "reports" / "figures" / "rule_baseline_confusion_matrix.png").exists()
    assert "models" in results
