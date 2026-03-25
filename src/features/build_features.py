"""Build the model-ready feature matrix."""

from __future__ import annotations

import pandas as pd

from src.config import TARGET_COLUMN
from src.data.preprocess import validate_schema
from src.features.rule_features import add_rule_features


def build_feature_matrix(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target labels for fraud-model experiments."""
    validate_schema(dataframe)

    features = add_rule_features(dataframe)
    target = features[TARGET_COLUMN].astype(int)

    model_frame = features[
        [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "is_full_balance_transfer",
            "is_full_balance_cash_out",
            "origin_balance_delta",
            "destination_balance_delta",
            "type",
        ]
    ].copy()
    model_frame = pd.get_dummies(model_frame, columns=["type"], dtype=int)
    return model_frame, target
