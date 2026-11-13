"""Load the saved pipeline and score new records.

Run from the project root:
    python -m src.predict                 # scores a built-in example
    python -m src.predict path/to/new.csv # scores a CSV with the same columns
"""
import sys

import joblib
import pandas as pd

from . import config

EXAMPLE = {
    "age": 45, "workclass": "Private", "education_num": 13,
    "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
    "relationship": "Husband", "race": "White", "sex": "Male",
    "capital_gain": 0, "capital_loss": 0, "hours_per_week": 50,
    "native_country": "United-States",
}


def predict(df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(config.MODEL_PATH)
    features = bundle["preprocessor"].transform(df)
    out = df.copy()
    out["prob_over_50k"] = bundle["model"].predict_proba(features)[:, 1].round(4)
    out["prediction"] = (out["prob_over_50k"] >= 0.5).map({True: ">50K", False: "<=50K"})
    return out


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.DataFrame([EXAMPLE])
    print(predict(df)[["prob_over_50k", "prediction"]].to_string())
