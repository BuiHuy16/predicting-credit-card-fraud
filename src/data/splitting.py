"""Stable, shared train/validation split utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .loading import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN

DEFAULT_VALIDATION_IDS = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "validation"
    / "validation_ids_seed42.csv"
)
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.20


@dataclass(frozen=True)
class DataSplit:
    """Features, labels and row IDs for the fixed split."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    train_ids: pd.Series
    validation_ids: pd.Series


def _as_integer_id(value: object) -> int:
    numeric = int(value)
    if numeric != value:
        raise ValueError(f"ID must be an integer value; received {value!r}.")
    return numeric


def create_validation_ids(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> list[int]:
    """Create a deterministic stratified holdout by hashing IDs within class.

    Sorting by SHA-256(seed, class, ID) makes the split independent of row order
    and ML library sampling implementations. Per-class holdout counts are rounded
    to the nearest integer, matching the checked-in validation ID artifact.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if id_column not in frame or target_column not in frame:
        raise ValueError(f"Frame must include {id_column!r} and {target_column!r}.")
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be present and unique.")
    if frame[target_column].isna().any() or not frame[target_column].isin([0, 1]).all():
        raise ValueError(f"{target_column} must contain only non-missing 0 and 1.")

    selected: list[int] = []
    for label, group in frame.groupby(target_column, sort=True):
        ids = [_as_integer_id(value) for value in group[id_column].tolist()]
        count = round(len(ids) * validation_fraction)
        if count < 1 or count >= len(ids):
            raise ValueError(f"Not enough rows in class {label!r} for this split.")
        label_int = int(label)
        ids.sort(
            key=lambda row_id: hashlib.sha256(
                f"{seed}|{label_int}|{row_id}".encode("ascii")
            ).digest()
        )
        selected.extend(ids[:count])
    return sorted(selected)


def validation_split_id(
    validation_ids: list[int] | pd.Series | np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> str:
    """Return a human-readable split ID containing a hash of sorted IDs."""
    ids = sorted(_as_integer_id(value) for value in validation_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("validation_ids contains duplicates.")
    canonical = "".join(f"{value}\n" for value in ids)
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()[:12]
    percent = int(round(validation_fraction * 100))
    return f"stratified-holdout-v1-seed{seed}-val{percent}-{digest}"


def load_validation_ids(path: str | Path = DEFAULT_VALIDATION_IDS) -> pd.Series:
    """Read the checked-in validation ID list and reject malformed artifacts."""
    ids_frame = pd.read_csv(path)
    if list(ids_frame.columns) != [ID_COLUMN]:
        raise ValueError(f"Split artifact must contain exactly one {ID_COLUMN!r} column.")
    ids = ids_frame[ID_COLUMN]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Validation IDs must be present and unique.")
    return ids


def load_split(
    frame: pd.DataFrame,
    validation_ids_path: str | Path = DEFAULT_VALIDATION_IDS,
) -> DataSplit:
    """Partition a validated train DataFrame using the shared ID artifact."""
    required = {ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Train DataFrame is missing required columns: {sorted(missing)}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be present and unique before splitting.")
    if frame[TARGET_COLUMN].isna().any() or not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only non-missing 0 and 1.")

    validation_ids = load_validation_ids(validation_ids_path)
    unknown_ids = set(validation_ids) - set(frame[ID_COLUMN])
    if unknown_ids:
        example = next(iter(unknown_ids))
        raise ValueError(f"Split artifact contains an ID absent from train: {example}.")

    mask = frame[ID_COLUMN].isin(validation_ids)
    train_part = frame.loc[~mask]
    validation_part = frame.loc[mask]
    if train_part.empty or validation_part.empty:
        raise ValueError("Both training and validation partitions must be non-empty.")
    if train_part[TARGET_COLUMN].nunique() != 2 or validation_part[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Both partitions must contain both target classes.")

    return DataSplit(
        X_train=train_part.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True),
        X_validation=validation_part.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True),
        y_train=train_part[TARGET_COLUMN].reset_index(drop=True),
        y_validation=validation_part[TARGET_COLUMN].reset_index(drop=True),
        train_ids=train_part[ID_COLUMN].reset_index(drop=True),
        validation_ids=validation_part[ID_COLUMN].reset_index(drop=True),
    )
