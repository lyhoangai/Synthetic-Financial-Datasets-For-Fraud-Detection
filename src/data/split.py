"""Deterministic dataset splitting helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COLUMN, TEST_RATIO, VALIDATION_RATIO
from src.data.preprocess import validate_schema


def split_dataset(
    dataframe: pd.DataFrame,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train, validation, and test partitions."""
    validate_schema(dataframe)

    train_frame, test_frame = train_test_split(
        dataframe,
        test_size=test_ratio,
        random_state=RANDOM_STATE,
        stratify=dataframe[TARGET_COLUMN],
    )
    adjusted_validation_ratio = validation_ratio / (1 - test_ratio)
    train_frame, validation_frame = train_test_split(
        train_frame,
        test_size=adjusted_validation_ratio,
        random_state=RANDOM_STATE,
        stratify=train_frame[TARGET_COLUMN],
    )

    return (
        train_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )
