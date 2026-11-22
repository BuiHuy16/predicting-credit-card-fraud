"""Loading and schema checks for the raw fraud dataset."""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

ID_COLUMN: Final = "id"
TARGET_COLUMN: Final = "is_fraud"

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
)
BINARY_FEATURES: Final[tuple[str, ...]] = (
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    *BINARY_FEATURES,
)
TRAIN_COLUMNS: Final[tuple[str, ...]] = (
    ID_COLUMN,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
)
TEST_COLUMNS: Final[tuple[str, ...]] = (ID_COLUMN, *FEATURE_COLUMNS)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "raw"


def _check_columns(frame: pd.DataFrame, expected: tuple[str, ...], name: str) -> None:
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{name} contains duplicate column names: {duplicates}")
    if tuple(frame.columns) != expected:
        missing = sorted(set(expected) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(expected))
        raise ValueError(
            f"{name} schema does not match. Expected ordered columns {expected}; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _check_features(frame: pd.DataFrame, name: str) -> None:
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{name}.{ID_COLUMN} must be present and unique.")

    wrong_numeric = [
        column
        for column in NUMERIC_FEATURES
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    if wrong_numeric:
        raise TypeError(f"Expected numerical columns in {name}: {wrong_numeric}")

    wrong_categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if not (
            pd.api.types.is_string_dtype(frame[column])
            or pd.api.types.is_object_dtype(frame[column])
        )
    ]
    if wrong_categorical:
        raise TypeError(f"Expected string columns in {name}: {wrong_categorical}")

    wrong_binary = [
        column
        for column in BINARY_FEATURES
        if not pd.api.types.is_bool_dtype(frame[column])
        and not frame[column]
        .dropna()
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .all()
    ]
    if wrong_binary:
        raise TypeError(f"Expected boolean columns in {name}: {wrong_binary}")

    for column in NUMERIC_FEATURES:
        values = frame[column].dropna().to_numpy()
        if values.size and not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} contains an infinite value.")


def validate_train(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a train DataFrame and return it unchanged.

    Feature missing values are allowed because the shared preprocessing pipeline
    imputes them. The target and ID may not be missing.
    """
    _check_columns(frame, TRAIN_COLUMNS, "train")
    _check_features(frame, "train")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"train.{TARGET_COLUMN} may not contain missing values.")
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"train.{TARGET_COLUMN} must contain only 0 and 1.")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("train must contain both target classes 0 and 1.")
    return frame


def validate_test(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a test DataFrame and return it unchanged."""
    _check_columns(frame, TEST_COLUMNS, "test")
    _check_features(frame, "test")
    return frame


def load_train(path: str | Path | None = None) -> pd.DataFrame:
    """Read and validate the raw training CSV."""
    csv_path = Path(path) if path is not None else RAW_DATA_DIR / "train.csv"
    return validate_train(pd.read_csv(csv_path))


def load_test(path: str | Path | None = None) -> pd.DataFrame:
    """Read and validate the raw test CSV."""
    csv_path = Path(path) if path is not None else RAW_DATA_DIR / "test.csv"
    return validate_test(pd.read_csv(csv_path))


def load_train_test(
    data_dir: str | Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both raw files and confirm their feature schemas and IDs agree."""
    data_dir = Path(data_dir)
    train = load_train(data_dir / "train.csv")
    test = load_test(data_dir / "test.csv")
    overlap = set(train[ID_COLUMN]) & set(test[ID_COLUMN])
    if overlap:
        raise ValueError(f"Train and test IDs overlap (example: {next(iter(overlap))}).")
    return train, test
