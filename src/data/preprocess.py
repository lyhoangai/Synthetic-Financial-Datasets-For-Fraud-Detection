"""Schema validation and basic preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_COLUMNS


def validate_schema(dataframe: pd.DataFrame) -> None:
    """Raise an error when required PaySim columns are missing."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")
