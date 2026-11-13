"""Hand-written preprocessing: imputation, scaling, rare-category grouping,
one-hot encoding.

Every statistic (medians, means, stds, modes, category lists) is learned in
fit() from the TRAINING data only, then reused in transform() for validation,
test, and new data. That is what prevents leakage.
"""
import pandas as pd

OTHER = "__other__"


def split_column_types(X: pd.DataFrame):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


class Preprocessor:
    def __init__(self, min_frequency: float = 0.01):
        # Categories rarer than this share of training rows are grouped into OTHER,
        # which keeps the one-hot feature count small (e.g. native_country).
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.numeric_, self.categorical_ = split_column_types(X)

        # Numeric: median imputation, then standardization.
        self.medians_ = X[self.numeric_].median()
        filled = X[self.numeric_].fillna(self.medians_)
        self.means_ = filled.mean()
        self.stds_ = filled.std(ddof=0).replace(0, 1.0)  # avoid divide-by-zero

        # Categorical: mode imputation, then the set of kept categories.
        self.modes_ = X[self.categorical_].mode().iloc[0]
        self.categories_ = {}
        for col in self.categorical_:
            freq = X[col].fillna(self.modes_[col]).value_counts(normalize=True)
            self.categories_[col] = sorted(freq[freq >= self.min_frequency].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = (X[self.numeric_].fillna(self.medians_) - self.means_) / self.stds_

        one_hot = {}
        for col in self.categorical_:
            values = X[col].fillna(self.modes_[col])
            # Rare or never-seen categories map to OTHER.
            values = values.where(values.isin(self.categories_[col]), OTHER)
            for category in self.categories_[col] + [OTHER]:
                one_hot[f"{col}={category}"] = (values == category).astype(float)

        out = pd.concat([numeric, pd.DataFrame(one_hot, index=X.index)], axis=1)
        return out.astype(float)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @property
    def feature_names(self):
        return self.numeric_ + [
            f"{col}={c}" for col in self.categorical_
            for c in self.categories_[col] + [OTHER]
        ]
