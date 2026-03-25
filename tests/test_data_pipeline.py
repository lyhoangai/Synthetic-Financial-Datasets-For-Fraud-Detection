from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_paysim_csv
from src.data.preprocess import validate_schema
from src.data.split import split_dataset


def _make_dataset(rows: int = 20) -> pd.DataFrame:
    labels = [0] * (rows - 4) + [1, 1, 1, 1]
    records = []
    for idx in range(rows):
        records.append(
            {
                "step": idx + 1,
                "type": "TRANSFER" if idx % 2 == 0 else "CASH_OUT",
                "amount": float(100 + idx),
                "nameOrig": f"C{idx:06d}",
                "oldbalanceOrg": float(100 + idx),
                "newbalanceOrig": float(idx % 3),
                "nameDest": f"M{idx:06d}",
                "oldbalanceDest": float(50 + idx),
                "newbalanceDest": float(60 + idx),
                "isFraud": labels[idx],
                "isFlaggedFraud": 0,
            }
        )
    return pd.DataFrame(records)


def test_validate_schema_accepts_expected_columns():
    validate_schema(_make_dataset(rows=3))


def test_validate_schema_rejects_missing_columns():
    df = _make_dataset(rows=3).drop(columns=["nameDest"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_load_paysim_csv_round_trips(tmp_path: Path):
    source = _make_dataset(rows=6)
    csv_path = tmp_path / "paysim_sample.csv"
    source.to_csv(csv_path, index=False)

    loaded = load_paysim_csv(csv_path)

    assert loaded.shape == source.shape
    assert loaded.columns.tolist() == source.columns.tolist()


def test_split_dataset_is_deterministic():
    df = _make_dataset(rows=20)

    first_train, first_val, first_test = split_dataset(df)
    second_train, second_val, second_test = split_dataset(df)

    assert first_train["nameOrig"].tolist() == second_train["nameOrig"].tolist()
    assert first_val["nameOrig"].tolist() == second_val["nameOrig"].tolist()
    assert first_test["nameOrig"].tolist() == second_test["nameOrig"].tolist()
