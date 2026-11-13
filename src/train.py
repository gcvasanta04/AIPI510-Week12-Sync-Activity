"""Preprocess, train all candidates, pick the best on validation, report on
test, save.

Run from the project root:  python -m src.train
"""
import json
import time

import joblib

from . import config
from .data import load_dataset
from .evaluate import evaluate, report
from .features import Preprocessor
from .models import candidate_models


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
          f"positive rate={y_train.mean():.3f}")

    # Fit preprocessing on training data ONLY, then apply it to val and test.
    preprocessor = Preprocessor().fit(X_train)
    Xt_train = preprocessor.transform(X_train)
    Xt_val = preprocessor.transform(X_val)
    Xt_test = preprocessor.transform(X_test)
    print(f"features after preprocessing: {Xt_train.shape[1]}")

    results, fitted = {}, {}
    for name, model in candidate_models().items():
        start = time.time()
        model.fit(Xt_train, y_train)
        results[name] = evaluate(model, Xt_val, y_val)
        fitted[name] = model
        print(f"{name:<24} val={results[name]}  ({time.time() - start:.1f}s)")

    # Select on validation ROC AUC; the test set is touched exactly once.
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = fitted[best_name]
    test_metrics = evaluate(best, Xt_test, y_test)

    print(f"\nBest model: {best_name}")
    print(f"Test metrics: {test_metrics}\n")
    print(report(best, Xt_test, y_test))

    # Save preprocessor and model together; inference needs both.
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": best}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(
        {"best_model": best_name, "validation": results, "test": test_metrics},
        indent=2,
    ))
    print(f"Saved preprocessor + model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
