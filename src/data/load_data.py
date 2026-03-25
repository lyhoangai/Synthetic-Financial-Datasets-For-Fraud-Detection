"""Utilities for loading the PaySim dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import validate_schema


def load_paysim_csv(path: str | Path) -> pd.DataFrame:
    """Load the PaySim CSV and validate that required columns exist."""
    csv_path = Path(path)
    dataframe = pd.read_csv(csv_path)
    validate_schema(dataframe)
    return dataframe
