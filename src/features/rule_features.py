"""Rule-inspired feature helpers."""

from __future__ import annotations

import pandas as pd


def add_rule_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add domain-inspired features from the PaySim fraud patterns."""
    features = dataframe.copy()
    features["is_full_balance_transfer"] = (
        (features["type"] == "TRANSFER")
        & (features["amount"] > 0)
        & (features["amount"] == features["oldbalanceOrg"])
    ).astype(int)
    features["is_full_balance_cash_out"] = (
        (features["type"] == "CASH_OUT")
        & (features["amount"] > 0)
        & (features["amount"] == features["oldbalanceOrg"])
    ).astype(int)
    features["origin_balance_delta"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )
    features["destination_balance_delta"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )
    return features
