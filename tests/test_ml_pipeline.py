"""Unit and smoke tests for ML pipelines, serialization, and evaluation metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loading import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from src.evaluation.metrics import compute_metrics
from src.models.ml import (
    HAS_CATBOOST,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    MLPipeline,
    build_model,
    make_tree_preprocessor,
)
from src.prediction.predict_ml import load_pipeline, predict_probabilities


def create_dummy_dataset(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data matching exact project schema."""
    rng = np.random.default_rng(seed)

    data: dict[str, Any] = {
        ID_COLUMN: np.arange(n_samples, dtype=int),
    }

    # 13 Numeric features
    for col in NUMERIC_FEATURES:
        data[col] = rng.normal(loc=10.0, scale=5.0, size=n_samples)

    # 5 Categorical features
    cat_options = {
        "merchant_category": ["grocery", "online", "travel", "entertainment"],
        "card_type": ["visa", "mastercard", "amex"],
        "auth_method": ["pin", "chip", "contactless", "biometric"],
        "channel": ["web", "mobile", "pos", "atm"],
        "device_type": ["ios", "android", "windows", "macos"],
    }
    for col in CATEGORICAL_FEATURES:
        options = cat_options.get(col, ["cat_a", "cat_b", "cat_c"])
        data[col] = rng.choice(options, size=n_samples)

    # 6 Binary features
    for col in BINARY_FEATURES:
        data[col] = rng.choice([False, True], size=n_samples)

    # Target: 90% class 0, 10% class 1
    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = rng.choice(n_samples, size=max(2, int(n_samples * 0.1)), replace=False)
    labels[fraud_indices] = 1
    data[TARGET_COLUMN] = labels

    return pd.DataFrame(data)


class TestMLPipeline(unittest.TestCase):
    """Test suite for ML model creation, fitting, prediction, and leakage prevention."""

    def setUp(self) -> None:
        self.df = create_dummy_dataset(n_samples=120, seed=42)
        # Split into 80 train / 40 val
        self.train_df = self.df.iloc[:80].copy()
        self.val_df = self.df.iloc[80:].copy()

        self.X_train = self.train_df.loc[:, list(FEATURE_COLUMNS)]
        self.y_train = self.train_df[TARGET_COLUMN]
        self.X_val = self.val_df.loc[:, list(FEATURE_COLUMNS)]
        self.y_val = self.val_df[TARGET_COLUMN]

    def test_logistic_regression_pipeline(self) -> None:
        """Test Logistic Regression fitting, probability range, and output shape."""
        pipeline = build_model("logistic_regression", params={"max_iter": 200}, random_seed=42)
        self.assertIsInstance(pipeline, MLPipeline)

        pipeline.fit(self.X_train, self.y_train)
        probs = pipeline.predict_fraud_probability(self.X_val)

        self.assertEqual(len(probs), len(self.val_df))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_random_forest_pipeline(self) -> None:
        """Test Random Forest tree preprocessor and probability predictions."""
        pipeline = build_model(
            "random_forest",
            params={"n_estimators": 10, "max_depth": 4},
            random_seed=42,
        )
        self.assertIsInstance(pipeline, MLPipeline)

        pipeline.fit(self.X_train, self.y_train)
        probs = pipeline.predict_fraud_probability(self.X_val)

        self.assertEqual(len(probs), len(self.val_df))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_tree_preprocessor_handles_unseen_categories(self) -> None:
        """Test that unseen categories in validation do not crash the tree preprocessor."""
        preprocessor = make_tree_preprocessor()
        preprocessor.fit(self.X_train)

        # Inject unseen category in validation
        X_val_unseen = self.X_val.copy()
        X_val_unseen["merchant_category"] = "unseen_rare_merchant_xyz"

        trans = preprocessor.transform(X_val_unseen)
        self.assertEqual(trans.shape[0], len(X_val_unseen))
        self.assertTrue(np.isfinite(trans).all())

    def test_gbdt_fallback_or_early_stopping(self) -> None:
        """Test GBDT models: early stopping if installed, otherwise clean ImportError."""
        gbdt_models = [
            ("xgboost", HAS_XGBOOST),
            ("lightgbm", HAS_LIGHTGBM),
            ("catboost", HAS_CATBOOST),
        ]
        for model_name, is_installed in gbdt_models:
            with self.subTest(model=model_name, installed=is_installed):
                if not is_installed:
                    with self.assertRaises(ImportError) as ctx:
                        build_model(model_name)
                    self.assertIn("not installed", str(ctx.exception).lower())
                else:
                    pipeline = build_model(
                        model_name,
                        params={"n_estimators": 15, "iterations": 15},
                        random_seed=42,
                    )
                    pipeline.fit(
                        self.X_train,
                        self.y_train,
                        X_val=self.X_val,
                        y_val=self.y_val,
                        early_stopping_rounds=5,
                    )
                    probs = pipeline.predict_fraud_probability(self.X_val)
                    self.assertEqual(len(probs), len(self.val_df))
                    self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_pipeline_serialization_and_inference(self) -> None:
        """Verify saved pipeline can be loaded and produces identical probabilities."""
        pipeline = build_model("random_forest", params={"n_estimators": 5}, random_seed=42)
        pipeline.fit(self.X_train, self.y_train)

        pred_before = pipeline.predict_fraud_probability(self.X_val)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "rf_test.joblib"
            joblib.dump(pipeline, artifact_path)

            loaded_pipeline = load_pipeline(artifact_path)
            pred_after = loaded_pipeline.predict_fraud_probability(self.X_val)

            np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-5)

            # Test predict_probabilities formatting
            pred_df = predict_probabilities(loaded_pipeline, self.val_df)
            self.assertIn(ID_COLUMN, pred_df.columns)
            self.assertIn("is_fraud", pred_df.columns)
            self.assertEqual(len(pred_df), len(self.val_df))

    def test_compute_metrics(self) -> None:
        """Test evaluation metrics (ROC-AUC, PR-AUC, threshold metrics)."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

        metrics = compute_metrics(y_true, y_score, threshold=0.5)

        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertGreaterEqual(metrics["pr_auc"], 0.9)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
