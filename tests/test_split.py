"""Reproducibility and partition tests for validation splits."""

import pandas as pd
import pytest

from src.data.loading import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data.splitting import (
    create_validation_ids,
    load_split,
    validation_split_id,
)


def make_frame(n: int = 50) -> pd.DataFrame:
    labels = [0] * (n - n // 5) + [1] * (n // 5)
    data = {"id": list(range(1000, 1000 + n)), "is_fraud": labels}
    data.update({column: [float(i) for i in range(n)] for column in NUMERIC_FEATURES})
    data.update({column: ["A" if i % 2 else "B" for i in range(n)] for column in CATEGORICAL_FEATURES})
    data.update({column: [bool(i % 2) for i in range(n)] for column in BINARY_FEATURES})
    return pd.DataFrame(data)


def test_create_validation_ids_is_deterministic_and_stratified() -> None:
    frame = make_frame()
    first = create_validation_ids(frame)
    second = create_validation_ids(frame.sample(frac=1, random_state=7))
    assert first == second
    assert len(first) == 10
    assert validation_split_id(first) == validation_split_id(second)


def test_split_excludes_id_and_target_from_features(tmp_path) -> None:
    frame = make_frame()
    ids = create_validation_ids(frame)
    path = tmp_path / "validation_ids.csv"
    pd.DataFrame({"id": ids}).to_csv(path, index=False)

    split = load_split(frame, path)
    assert len(split.X_train) == 40
    assert len(split.X_validation) == 10
    assert "id" not in split.X_train.columns
    assert "is_fraud" not in split.X_train.columns
    assert set(split.train_ids).isdisjoint(set(split.validation_ids))
    assert set(split.train_ids) | set(split.validation_ids) == set(frame["id"])
    assert set(split.y_train) == {0, 1}
    assert set(split.y_validation) == {0, 1}
    assert len(split.y_validation) == 10


def test_load_split_rejects_unknown_validation_id(tmp_path) -> None:
    frame = make_frame()
    path = tmp_path / "validation_ids.csv"
    pd.DataFrame({"id": [999999]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="absent from train"):
        load_split(frame, path)
