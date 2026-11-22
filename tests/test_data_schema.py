"""Schema validation tests for the raw fraud dataset."""

import pandas as pd
import pytest

from src.data.loading import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TRAIN_COLUMNS,
    TEST_COLUMNS,
    validate_test,
    validate_train,
)


def make_frame(*, include_target: bool = True) -> pd.DataFrame:
    data = {"id": [10, 11, 12, 13]}
    data.update({column: [1.0, 2.0, 3.0, 4.0] for column in NUMERIC_FEATURES})
    data.update({column: ["A", "B", "A", "B"] for column in CATEGORICAL_FEATURES})
    data.update({column: [False, True, False, True] for column in BINARY_FEATURES})
    if include_target:
        data[TARGET_COLUMN] = [0, 1, 0, 1]
        return pd.DataFrame(data).loc[:, list(TRAIN_COLUMNS)]
    return pd.DataFrame(data).loc[:, list(TEST_COLUMNS)]


def test_validate_train_accepts_expected_schema() -> None:
    frame = make_frame()
    assert validate_train(frame) is frame


def test_validate_test_accepts_expected_schema() -> None:
    frame = make_frame(include_target=False)
    assert validate_test(frame) is frame


def test_validate_train_rejects_invalid_target() -> None:
    frame = make_frame()
    frame.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_train(frame)


def test_validate_train_rejects_duplicate_ids() -> None:
    frame = make_frame()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="must be present and unique"):
        validate_train(frame)


def test_validate_test_rejects_target_column() -> None:
    frame = make_frame()
    with pytest.raises(ValueError, match="schema does not match"):
        validate_test(frame)
