"""Shared preprocessing factories for ML and neural-network pipelines."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .loading import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _boolean_to_float32(values: object) -> np.ndarray:
    """Convert bool/nullable-bool columns to numeric 0/1 with NaN preserved."""
    return pd.DataFrame(values).astype("boolean").to_numpy(
        dtype=np.float32,
        na_value=np.nan,
    )


def make_preprocessor(
    *,
    scale_numeric: bool = True,
    sparse_threshold: float = 0.3,
) -> ColumnTransformer:
    """Build a transformer that must be fit on training rows only.

    Numerical columns are median-imputed and optionally standardized. String categories
    are mode-imputed and one-hot encoded with safe handling of unseen values.
    Boolean columns are mode-imputed and kept as 0/1-like features without
    scaling. The returned transformer does not include ID or target.
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=numeric_steps)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    binary = Pipeline(
        steps=[
            (
                "to_numeric",
                FunctionTransformer(
                    _boolean_to_float32,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("binary", binary, list(BINARY_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=False,
    )
