"""Preprocessing contract tests."""

import numpy as np
import pandas as pd

from src.data.loading import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data.preprocessing import make_preprocessor


def make_features(n: int = 12) -> pd.DataFrame:
    data = {column: np.arange(n, dtype=float) for column in NUMERIC_FEATURES}
    data.update({column: ["A" if i % 2 else "B" for i in range(n)] for column in CATEGORICAL_FEATURES})
    data.update({column: [bool(i % 2) for i in range(n)] for column in BINARY_FEATURES})
    return pd.DataFrame(data)


def test_preprocessor_fits_on_training_rows_and_accepts_unknown_categories() -> None:
    train = make_features()
    validation = make_features(3).copy()
    validation.loc[:, list(CATEGORICAL_FEATURES)] = "unseen-category"
    validation.loc[:, list(NUMERIC_FEATURES)] = 10_000.0

    preprocessor = make_preprocessor()
    train_transformed = preprocessor.fit_transform(train)
    validation_transformed = preprocessor.transform(validation)

    expected_means = train.loc[:, list(NUMERIC_FEATURES)].mean().to_numpy()
    actual_means = preprocessor.named_transformers_["numeric"].named_steps["scaler"].mean_
    np.testing.assert_allclose(actual_means, expected_means)
    assert train_transformed.shape[0] == len(train)
    assert validation_transformed.shape[0] == len(validation)
    assert train_transformed.shape[1] == validation_transformed.shape[1]

    train_values = train_transformed.toarray() if hasattr(train_transformed, "toarray") else train_transformed
    validation_values = (
        validation_transformed.toarray()
        if hasattr(validation_transformed, "toarray")
        else validation_transformed
    )
    assert np.isfinite(train_values).all()
    assert np.isfinite(validation_values).all()
