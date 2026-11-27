from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loading import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    load_test,
)
from src.models.mlp import FraudMLP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "dl_mlp_seed42_baseline_v1"
RUN_DIR = PROJECT_ROOT / "experiments" / "dl" / RUN_ID
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / f"{RUN_ID}.pt"
PREPROCESSOR_PATH = RUN_DIR / "preprocessor.joblib"
CONFIG_PATH = RUN_DIR / "config.yaml"
TEST_PATH = PROJECT_ROOT / "datasets" / "raw" / "test.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "datasets" / "raw" / "sample_submission.csv"
SUBMISSION_PATH = PROJECT_ROOT / "outputs" / "submissions" / "submission.csv"
DEFAULT_BATCH_SIZE = 8192


def _load_baseline(
    checkpoint_path: Path,
    preprocessor_path: Path,
    config_path: Path,
) -> tuple[FraudMLP, Any, dict[str, Any]]:
    device = torch.device("cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if checkpoint.get("run_id") != RUN_ID or config.get("run_id") != RUN_ID:
        raise ValueError(f"Expected baseline run_id {RUN_ID!r}.")

    excluded = checkpoint.get("temporarily_excluded_features")
    if excluded != config["preprocessing"]["temporarily_excluded_features"]:
        raise ValueError("Checkpoint and config disagree on excluded features.")
    expected_feature_order = [column for column in FEATURE_COLUMNS if column not in excluded]
    feature_order = checkpoint.get("feature_order")
    if feature_order != expected_feature_order:
        raise ValueError("Checkpoint feature order differs from the DL baseline policy.")
    if config["runtime"]["feature_order"] != expected_feature_order:
        raise ValueError("Saved config feature order differs from the DL baseline policy.")

    architecture = checkpoint["architecture"]
    if architecture["input_dim"] != config["runtime"]["input_dim"]:
        raise ValueError("Checkpoint and config input dimensions differ.")
    if architecture["hidden_dims"] != config["model"]["hidden_dims"]:
        raise ValueError("Checkpoint and config hidden dimensions differ.")
    if architecture["dropout"] != config["model"]["dropout"]:
        raise ValueError("Checkpoint and config dropout values differ.")

    preprocessor = joblib.load(preprocessor_path)
    if list(preprocessor.feature_names_in_) != expected_feature_order:
        raise ValueError("Saved preprocessor feature order differs from the DL baseline.")
    if len(preprocessor.get_feature_names_out()) != architecture["input_dim"]:
        raise ValueError("Saved preprocessor output dimension differs from the checkpoint.")

    model = FraudMLP(
        input_dim=architecture["input_dim"],
        hidden_dims=tuple(architecture["hidden_dims"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, preprocessor, checkpoint


def _dense_float32(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Preprocessed test features contain NaN or Inf.")
    return np.ascontiguousarray(array)


def predict_probabilities(
    model: FraudMLP,
    features: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size])
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))

    result = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Inference returned invalid fraud probabilities.")
    return result


def build_submission(
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    expected_columns = [ID_COLUMN, TARGET_COLUMN]
    if list(sample_submission.columns) != expected_columns:
        raise ValueError(
            f"Sample submission columns must be {expected_columns}; "
            f"got {list(sample_submission.columns)}."
        )
    if test[ID_COLUMN].isna().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("Test IDs must be present and unique.")
    if sample_submission[ID_COLUMN].isna().any() or sample_submission[ID_COLUMN].duplicated().any():
        raise ValueError("Sample submission IDs must be present and unique.")
    if not test[ID_COLUMN].reset_index(drop=True).equals(
        sample_submission[ID_COLUMN].reset_index(drop=True)
    ):
        raise ValueError("Test and sample submission IDs differ in value or order.")
    if probabilities.ndim != 1 or len(probabilities) != len(test):
        raise ValueError("Probability count must equal the test row count.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Fraud probabilities must be finite values in [0, 1].")

    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(copy=True),
            TARGET_COLUMN: probabilities,
        }
    )
    validate_submission(submission, sample_submission)
    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Submission schema or column order differs from the sample.")
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count differs from the sample submission.")
    if submission[ID_COLUMN].isna().any() or submission[ID_COLUMN].duplicated().any():
        raise ValueError("Submission IDs must be present and unique.")
    if not submission[ID_COLUMN].reset_index(drop=True).equals(
        sample_submission[ID_COLUMN].reset_index(drop=True)
    ):
        raise ValueError("Submission IDs differ from the sample in value or order.")

    probabilities = pd.to_numeric(submission[TARGET_COLUMN], errors="raise").to_numpy()
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Submission probabilities must be finite values in [0, 1].")


def run_inference(
    *,
    test_path: Path = TEST_PATH,
    sample_submission_path: Path = SAMPLE_SUBMISSION_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    config_path: Path = CONFIG_PATH,
    output_path: Path = SUBMISSION_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    test = load_test(test_path)
    sample_submission = pd.read_csv(sample_submission_path)
    model, preprocessor, checkpoint = _load_baseline(
        checkpoint_path,
        preprocessor_path,
        config_path,
    )
    feature_order = checkpoint["feature_order"]
    transformed = _dense_float32(preprocessor.transform(test.loc[:, feature_order]))
    if transformed.shape[1] != checkpoint["architecture"]["input_dim"]:
        raise ValueError("Transformed test feature count differs from the checkpoint.")

    probabilities = predict_probabilities(model, transformed, batch_size=batch_size)
    submission = build_submission(test, sample_submission, probabilities)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    saved_submission = pd.read_csv(output_path)
    validate_submission(saved_submission, sample_submission)
    return {
        "run_id": checkpoint["run_id"],
        "split_id": checkpoint["split_id"],
        "rows": len(saved_submission),
        "columns": list(saved_submission.columns),
        "probability_min": float(saved_submission[TARGET_COLUMN].min()),
        "probability_max": float(saved_submission[TARGET_COLUMN].max()),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DL baseline test submission.")
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--sample-submission", type=Path, default=SAMPLE_SUBMISSION_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--preprocessor", type=Path, default=PREPROCESSOR_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    print(
        json.dumps(
            run_inference(
                test_path=args.test,
                sample_submission_path=args.sample_submission,
                checkpoint_path=args.checkpoint,
                preprocessor_path=args.preprocessor,
                config_path=args.config,
                output_path=args.output,
                batch_size=args.batch_size,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
