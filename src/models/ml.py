"""Machine Learning models and factory for Credit Card Fraud Detection."""

from __future__ import annotations

import logging
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from ..data.loading import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ..data.preprocessing import _boolean_to_float32
from .ml_features import (
    ALL_NUMERIC_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    MLFeatureEngineer,
    get_numeric_features,
)

logger = logging.getLogger(__name__)

# Fallback imports for optional GBDT packages
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    LGBMClassifier = None  # type: ignore
    lgb_early_stopping = None  # type: ignore

try:
    import catboost as cb
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoostClassifier = None  # type: ignore


# Approximate ratio of majority (0) to minority (1) class in train (~550,617 / 9,351)
DEFAULT_SCALE_POS_WEIGHT = 58.88


def make_linear_preprocessor(
    numeric_features: Sequence[str] | None = None,
    scale_numeric: bool = True,
    sparse_threshold: float = 0.3,
) -> ColumnTransformer:
    """Build a preprocessor specifically designed for linear models (Logistic Regression).

    - Numeric features: Median-imputed, StandardScaled (automatically includes engineered features).
    - Categorical features: Mode-imputed, OneHotEncoded (handles unseen values safely).
    - Binary features: Converted to float32 (0.0 / 1.0), mode-imputed.
    """
    num_selector = list(numeric_features) if numeric_features is not None else get_numeric_features

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
            ("numeric", numeric, num_selector),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("binary", binary, list(BINARY_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=False,
    )


def make_tree_preprocessor(
    numeric_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build a preprocessor specifically designed for tree-based models.

    - Numeric features: Median-imputed, no scaling (automatically includes engineered features).
    - Categorical features: Mode-imputed, OrdinalEncoded (handles unseen values safely).
    - Binary features: Converted to float32 (0.0 / 1.0), mode-imputed.
    Outputs a dense 2D array suitable for Random Forest, XGBoost, LightGBM, and CatBoost.
    """
    num_selector = list(numeric_features) if numeric_features is not None else get_numeric_features

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
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
            ("numeric", numeric, num_selector),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("binary", binary, list(BINARY_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


class MLPipeline:
    """Unified wrapper combining feature engineering, preprocessing, and an estimator.

    Guarantees 100% no data leakage by fitting both MLFeatureEngineer and the preprocessor
    ONLY on the training split.
    """

    def __init__(
        self,
        model_name: str,
        preprocessor: ColumnTransformer,
        estimator: Any,
        is_gbdt: bool = False,
        feature_engineer: MLFeatureEngineer | None = None,
    ) -> None:
        self.model_name = model_name
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.is_gbdt = is_gbdt
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
    ) -> MLPipeline:
        """Fit feature engineer & preprocessor only on X_train, transform datasets, and fit estimator."""
        # 1. Feature Engineering: strictly fit on X_train only
        if self.feature_engineer is not None:
            self.feature_engineer.fit(X_train)
            X_train = self.feature_engineer.transform(X_train)
            if X_val is not None:
                X_val = self.feature_engineer.transform(X_val)

        # 2. Fit preprocessor strictly on X_train
        self.preprocessor.fit(X_train)
        X_train_trans = self.preprocessor.transform(X_train)

        # 3. Fit estimator (with early stopping if GBDT and val set provided)
        if self.is_gbdt and X_val is not None and y_val is not None and early_stopping_rounds:
            X_val_trans = self.preprocessor.transform(X_val)

            if self.model_name == "xgboost":
                self.estimator.set_params(early_stopping_rounds=early_stopping_rounds)
                self.estimator.fit(
                    X_train_trans,
                    y_train,
                    eval_set=[(X_val_trans, y_val)],
                    verbose=False,
                )
            elif self.model_name == "lightgbm":
                callbacks = [
                    lgb_early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                ]
                self.estimator.fit(
                    X_train_trans,
                    y_train,
                    eval_set=[(X_val_trans, y_val)],
                    eval_metric="auc",
                    callbacks=callbacks,
                )
            elif self.model_name == "catboost":
                self.estimator.fit(
                    X_train_trans,
                    y_train,
                    eval_set=(X_val_trans, y_val),
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            else:
                self.estimator.fit(X_train_trans, y_train)
        else:
            self.estimator.fit(X_train_trans, y_train)

        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features and predict class probabilities [P(y=0), P(y=1)]."""
        if self.feature_engineer is not None:
            X = self.feature_engineer.transform(X)
        X_trans = self.preprocessor.transform(X)
        return self.estimator.predict_proba(X_trans)

    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1D array of predicted probabilities for positive class (is_fraud=1)."""
        proba = self.predict_proba(X)
        if hasattr(self.estimator, "classes_"):
            classes = list(self.estimator.classes_)
            if 1 in classes:
                return proba[:, classes.index(1)]
        return proba[:, 1]


def build_model(
    model_name: str,
    params: dict[str, Any] | None = None,
    handle_imbalance: bool = False,
    use_feature_engineering: bool = True,
    random_seed: int = 42,
) -> MLPipeline:
    """Factory creating an MLPipeline with appropriate feature engineer, preprocessor, and estimator.

    Supported model names:
    - 'logistic_regression' (P0)
    - 'random_forest' (P0)
    - 'xgboost' (P0)
    - 'lightgbm' (P1)
    - 'catboost' (P1)
    """
    model_name = model_name.lower()
    user_params = dict(params or {})
    feature_engineer = MLFeatureEngineer() if use_feature_engineering else None

    if model_name == "logistic_regression":
        preprocessor = make_linear_preprocessor(scale_numeric=True)
        default_params: dict[str, Any] = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": random_seed,
        }
        if handle_imbalance:
            default_params["class_weight"] = "balanced"
        default_params.update(user_params)
        estimator = LogisticRegression(**default_params)
        return MLPipeline(
            model_name,
            preprocessor,
            estimator,
            is_gbdt=False,
            feature_engineer=feature_engineer,
        )

    elif model_name == "random_forest":
        preprocessor = make_tree_preprocessor()
        default_params = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": random_seed,
        }
        if handle_imbalance:
            default_params["class_weight"] = "balanced_subsample"
        default_params.update(user_params)
        estimator = RandomForestClassifier(**default_params)
        return MLPipeline(
            model_name,
            preprocessor,
            estimator,
            is_gbdt=False,
            feature_engineer=feature_engineer,
        )

    elif model_name == "xgboost":
        if not HAS_XGBOOST or XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Please install it using 'pip install xgboost'."
            )
        preprocessor = make_tree_preprocessor()
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "random_state": random_seed,
            "n_jobs": -1,
        }
        if "iterations" in user_params:
            user_params.setdefault("n_estimators", user_params.pop("iterations"))
        if handle_imbalance and "scale_pos_weight" not in user_params:
            default_params["scale_pos_weight"] = DEFAULT_SCALE_POS_WEIGHT
        default_params.update(user_params)
        estimator = XGBClassifier(**default_params)
        return MLPipeline(
            model_name,
            preprocessor,
            estimator,
            is_gbdt=True,
            feature_engineer=feature_engineer,
        )

    elif model_name == "lightgbm":
        if not HAS_LIGHTGBM or LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Please install it using 'pip install lightgbm'."
            )
        preprocessor = make_tree_preprocessor()
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "metric": "auc",
            "random_state": random_seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        if "iterations" in user_params:
            user_params.setdefault("n_estimators", user_params.pop("iterations"))
        if handle_imbalance and "scale_pos_weight" not in user_params:
            default_params["scale_pos_weight"] = DEFAULT_SCALE_POS_WEIGHT
        default_params.update(user_params)
        estimator = LGBMClassifier(**default_params)
        return MLPipeline(
            model_name,
            preprocessor,
            estimator,
            is_gbdt=True,
            feature_engineer=feature_engineer,
        )

    elif model_name == "catboost":
        if not HAS_CATBOOST or CatBoostClassifier is None:
            raise ImportError(
                "catboost is not installed. Please install it using 'pip install catboost'."
            )
        preprocessor = make_tree_preprocessor()
        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "eval_metric": "AUC",
            "random_seed": random_seed,
            "verbose": 0,
        }
        # Harmonize parameter synonyms: if user passes n_estimators, drop iterations
        if "n_estimators" in user_params:
            default_params.pop("iterations", None)
        if handle_imbalance and "auto_class_weights" not in user_params:
            default_params["auto_class_weights"] = "Balanced"
        default_params.update(user_params)
        estimator = CatBoostClassifier(**default_params)
        return MLPipeline(
            model_name,
            preprocessor,
            estimator,
            is_gbdt=True,
            feature_engineer=feature_engineer,
        )

    else:
        supported = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
        ]
        raise ValueError(
            f"Unsupported model_name: {model_name!r}. Supported models: {supported}"
        )
