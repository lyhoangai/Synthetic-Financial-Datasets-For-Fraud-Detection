import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.inspection import extract_feature_importances


def test_extract_feature_importances_sorts_scores_descending():
    features = pd.DataFrame(
        {
            "signal": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "noise": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    target = [0, 0, 0, 1, 1, 1]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(features, target)

    importances = extract_feature_importances(model, features.columns.tolist())

    assert importances[0]["feature"] == "signal"
    assert importances[0]["importance"] >= importances[1]["importance"] >= 0.0
