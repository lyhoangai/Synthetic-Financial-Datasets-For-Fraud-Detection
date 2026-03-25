"""Project-wide configuration constants."""

from __future__ import annotations

RANDOM_STATE = 42
TARGET_COLUMN = "isFraud"
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}
