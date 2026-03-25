import pandas as pd

from src.features.build_features import build_feature_matrix


def test_build_feature_matrix_adds_rule_signal_columns():
    df = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [100.0, 120.0],
            "nameOrig": ["C000001", "C000002"],
            "oldbalanceOrg": [100.0, 120.0],
            "newbalanceOrig": [0.0, 10.0],
            "nameDest": ["M000001", "M000002"],
            "oldbalanceDest": [0.0, 20.0],
            "newbalanceDest": [100.0, 30.0],
            "isFraud": [1, 0],
            "isFlaggedFraud": [0, 0],
        }
    )

    features, target = build_feature_matrix(df)

    assert "is_full_balance_transfer" in features.columns
    assert "is_full_balance_cash_out" in features.columns
    assert "origin_balance_delta" in features.columns
    assert "destination_balance_delta" in features.columns
    assert "type_TRANSFER" in features.columns
    assert "type_CASH_OUT" in features.columns
    assert target.tolist() == [1, 0]
