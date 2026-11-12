"""Loading, cleaning, and splitting the Adult Census Income data."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
    return df.rename(columns={"class": config.TARGET})


def load_raw() -> pd.DataFrame:
    """Load from a local UCI file if present, otherwise from OpenML."""
    if config.LOCAL_RAW_PATH.exists():
        df = pd.read_csv(
            config.LOCAL_RAW_PATH, header=None, names=config.UCI_COLUMNS,
            skipinitialspace=True,
        )
    else:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("adult", version=2, as_frame=True).frame
    return _normalize_columns(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip whitespace in string columns and turn "?" into real missing values.
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    # Labels sometimes carry a trailing "." (UCI test file); normalize to 0/1.
    df[config.TARGET] = (
        df[config.TARGET].str.rstrip(".") == config.POSITIVE_LABEL
    ).astype(int)
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])
    return df.drop_duplicates().reset_index(drop=True)


def split(df: pd.DataFrame):
    """Stratified train / validation / test split."""
    X = df.drop(columns=config.TARGET)
    y = df[config.TARGET]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=config.VAL_SIZE, stratify=y_rest,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_dataset():
    return split(clean(load_raw()))
