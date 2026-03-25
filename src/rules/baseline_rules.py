"""Rule-based fraud baselines derived from PaySim analysis."""

from __future__ import annotations

import pandas as pd


def apply_rule_baseline(dataframe: pd.DataFrame) -> pd.Series:
    """Flag transactions that empty the origin balance via transfer or cash out."""
    full_balance_transfer = (
        (dataframe["type"] == "TRANSFER")
        & (dataframe["amount"] > 0)
        & (dataframe["amount"] == dataframe["oldbalanceOrg"])
    )
    full_balance_cash_out = (
        (dataframe["type"] == "CASH_OUT")
        & (dataframe["amount"] > 0)
        & (dataframe["amount"] == dataframe["oldbalanceOrg"])
    )
    return (full_balance_transfer | full_balance_cash_out).astype(int)
