"""Training script for Machine Learning models on the Credit Card Fraud dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

import joblib
import pandas as pd
import yaml

from ..data.loading import FEATURE_COLUMNS, ID_COLUMN, load_test, load_train
from ..data.splitting import load_split, validation_split_id
from ..evaluation.metrics import compute_metrics
from ..models.ml import build_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_ml")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ml.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def train_single_model(
    model_name: str,
    config: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train, evaluate, and save artifacts for a single ML model."""
    random_seed = int(config.get("random_seed", 42))
    model_cfg = config.get("models", {}).get(model_name, {})
    if not model_cfg:
        logger.warning("No specific config for %s; using default parameters.", model_name)

    handle_imbalance = bool(model_cfg.get("handle_imbalance", True))
    params = dict(model_cfg.get("params", {}))
    early_stopping_rounds = model_cfg.get("early_stopping_rounds", None)
    eval_cfg = config.get("evaluation", {})
    threshold = float(eval_cfg.get("default_threshold", 0.50))

    paths_cfg = config.get("paths", {})
    val_ids_path = PROJECT_ROOT / paths_cfg.get(
        "validation_ids", "experiments/validation/validation_ids_seed42.csv"
    )
    models_dir = PROJECT_ROOT / paths_cfg.get("model_output_dir", "outputs/models")
    val_preds_dir = PROJECT_ROOT / paths_cfg.get(
        "validation_predictions_dir", "outputs/predictions/validation"
    )
    test_preds_dir = PROJECT_ROOT / paths_cfg.get(
        "test_predictions_dir", "outputs/submissions"
    )
    results_csv_path = PROJECT_ROOT / paths_cfg.get(
        "results_csv", "experiments/ml/results.csv"
    )
    manifests_dir = PROJECT_ROOT / paths_cfg.get(
        "run_manifests_dir", "experiments/ml/run_manifests"
    )

    for d in (models_dir, val_preds_dir, test_preds_dir, manifests_dir, results_csv_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load split strictly adhering to validation protocol (80/20 seed 42)
    logger.info("Loading train/val split from artifact %s", val_ids_path)
    split = load_split(train_df, validation_ids_path=val_ids_path)
    split_id = validation_split_id(split.validation_ids, seed=random_seed)

    # 2. Build model pipeline
    logger.info("Building model pipeline for: %s", model_name)
    pipeline = build_model(
        model_name=model_name,
        params=params,
        handle_imbalance=handle_imbalance,
        random_seed=random_seed,
    )

    # 3. Fit pipeline on train partition only (with validation early-stopping for GBDT)
    logger.info("Starting training for %s...", model_name)
    start_time = time.time()
    pipeline.fit(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_validation,
        y_val=split.y_validation,
        early_stopping_rounds=early_stopping_rounds,
    )
    train_duration = time.time() - start_time
    logger.info("Training completed in %.2f seconds.", train_duration)

    # 4. Predict validation probabilities and compute metrics
    logger.info("Evaluating on validation partition (%d rows)...", len(split.validation_ids))
    val_probs = pipeline.predict_fraud_probability(split.X_validation)
    metrics = compute_metrics(split.y_validation, val_probs, threshold=threshold)
    logger.info(
        "Validation Metrics [%s] -> ROC-AUC: %.6f | PR-AUC: %.6f | F1 (th=%.2f): %.4f",
        model_name,
        metrics["roc_auc"],
        metrics["pr_auc"],
        threshold,
        metrics["f1"],
    )

    # 5. Predict on test partition
    logger.info("Predicting on test partition (%d rows)...", len(test_df))
    test_features = test_df.loc[:, list(FEATURE_COLUMNS)]
    test_probs = pipeline.predict_fraud_probability(test_features)

    # 6. Generate run identifier and save artifacts
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp_str}_{model_name}_seed{random_seed}"

    # 6.1 Save model artifact
    model_artifact_path = models_dir / f"{run_id}.joblib"
    joblib.dump(pipeline, model_artifact_path)
    logger.info("Model pipeline saved to: %s", model_artifact_path)

    # 6.2 Save validation predictions (id, y_true, is_fraud)
    val_pred_df = pd.DataFrame(
        {
            ID_COLUMN: split.validation_ids,
            "y_true": split.y_validation,
            "is_fraud": val_probs,
        }
    )
    val_pred_path = val_preds_dir / f"{run_id}_val_predictions.csv"
    val_pred_df.to_csv(val_pred_path, index=False)
    logger.info("Validation predictions saved to: %s", val_pred_path)

    # 6.3 Save test predictions formatted for Kaggle submission (id, is_fraud)
    test_pred_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            "is_fraud": test_probs,
        }
    )
    test_pred_path = test_preds_dir / f"{run_id}_submission.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    logger.info("Test submission saved to: %s", test_pred_path)

    # 6.4 Save run manifest JSON
    manifest_data = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "model": model_name,
        "split_id": split_id,
        "random_seed": random_seed,
        "handle_imbalance": handle_imbalance,
        "params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "train_time_seconds": round(train_duration, 3),
        "metrics": metrics,
        "artifacts": {
            "model_path": str(model_artifact_path.relative_to(PROJECT_ROOT)),
            "validation_predictions_path": str(val_pred_path.relative_to(PROJECT_ROOT)),
            "test_predictions_path": str(test_pred_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = manifests_dir / f"{run_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Run manifest saved to: %s", manifest_path)

    # 6.5 Append summary to results.csv
    result_row = {
        "run_id": run_id,
        "timestamp": timestamp_str,
        "model": model_name,
        "split_id": split_id,
        "seed": random_seed,
        "handle_imbalance": handle_imbalance,
        "roc_auc": round(metrics["roc_auc"], 6),
        "pr_auc": round(metrics["pr_auc"], 6),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "train_time_sec": round(train_duration, 2),
        "artifact_path": str(model_artifact_path.relative_to(PROJECT_ROOT)),
        "val_pred_path": str(val_pred_path.relative_to(PROJECT_ROOT)),
        "test_pred_path": str(test_pred_path.relative_to(PROJECT_ROOT)),
    }
    result_df = pd.DataFrame([result_row])
    write_header = not results_csv_path.is_file()
    result_df.to_csv(results_csv_path, mode="a", header=write_header, index=False)
    logger.info("Appended experiment summary to: %s", results_csv_path)

    return manifest_data


def run_training(
    models_to_run: list[str],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Run training workflow for specified models."""
    config = load_config(config_path)

    logger.info("Reading raw dataset...")
    train_df = load_train()
    test_df = load_test()
    logger.info("Dataset loaded successfully: Train=%s, Test=%s", train_df.shape, test_df.shape)

    results = []
    for model_name in models_to_run:
        logger.info("=== Running pipeline for: %s ===", model_name)
        try:
            manifest = train_single_model(model_name, config, train_df, test_df)
            results.append(manifest)
        except Exception as exc:
            logger.error("Failed training for model %s: %s", model_name, exc, exc_info=True)
            raise

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for Credit Card Fraud Detection.")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        help="Model to train: 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'p0', or 'all'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    model_arg = args.model.lower()
    if model_arg == "p0":
        models_to_run = ["logistic_regression", "random_forest", "xgboost"]
    elif model_arg == "all":
        models_to_run = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
    else:
        models_to_run = [model_arg]

    run_training(models_to_run=models_to_run, config_path=args.config)


if __name__ == "__main__":
    main()
