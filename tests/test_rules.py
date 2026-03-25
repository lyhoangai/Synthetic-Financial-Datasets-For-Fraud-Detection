import pandas as pd

from src.rules.baseline_rules import apply_rule_baseline


def test_rule_baseline_flags_full_balance_transfer():
    df = pd.DataFrame(
        {
            "type": ["TRANSFER"],
            "amount": [500.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [0.0],
        }
    )

    result = apply_rule_baseline(df)

    assert result.tolist() == [1]


def test_rule_baseline_flags_full_balance_cash_out():
    df = pd.DataFrame(
        {
            "type": ["CASH_OUT"],
            "amount": [200.0],
            "oldbalanceOrg": [200.0],
            "newbalanceOrig": [0.0],
        }
    )

    result = apply_rule_baseline(df)

    assert result.tolist() == [1]


def test_rule_baseline_ignores_normal_payment():
    df = pd.DataFrame(
        {
            "type": ["PAYMENT"],
            "amount": [50.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [450.0],
        }
    )

    result = apply_rule_baseline(df)

    assert result.tolist() == [0]
