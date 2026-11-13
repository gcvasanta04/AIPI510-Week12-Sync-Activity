"""Candidate models. They receive already-preprocessed features."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import config


def candidate_models():
    """Sensible defaults, no tuning. LR is the baseline to beat."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=config.RANDOM_STATE
        ),
    }
