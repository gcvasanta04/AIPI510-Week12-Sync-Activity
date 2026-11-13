"""Metrics. The classes are imbalanced (~24% positive), so accuracy alone
is misleading: report F1, ROC AUC, and PR AUC as well.

All functions expect X that has already gone through the Preprocessor."""
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    f1_score, roc_auc_score,
)


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
    }


def report(model, X, y) -> str:
    return classification_report(
        y, model.predict(X), target_names=["<=50K", ">50K"], digits=3
    )
