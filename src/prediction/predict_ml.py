"""Inference and submission generation utilities for ML pipelines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.loading import FEATURE_COLUMNS, ID_COLUMN, load_test
from ..models.ml import MLPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("predict_ml")


def load_pipeline(model_path: str | Path) -> MLPipeline:
    """Load a serialized MLPipeline artifact from disk."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    pipeline = joblib.load(path)
    if not isinstance(pipeline, MLPipeline):
        logger.warning(
            "Loaded object is of type %s, expected MLPipeline.", type(pipeline).__name__
        )
    return pipeline


def predict_probabilities(
    pipeline: MLPipeline,
    frame: pd.DataFrame,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Predict positive class probabilities (is_fraud) for a DataFrame.

    Returns a DataFrame with columns [id, is_fraud] matching the input row order.
    """
    if id_column not in frame.columns:
        raise ValueError(f"Input DataFrame must contain id column: {id_column!r}")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    probabilities = pipeline.predict_fraud_probability(features)

    # Validate output probabilities
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain NaN or infinite values.")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("Predicted probabilities fall outside the [0, 1] range.")

    return pd.DataFrame(
        {
            id_column: frame[id_column].values,
            "is_fraud": probabilities,
        }
    )


def generate_submission(
    model_path: str | Path,
    test_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a Kaggle-compliant submission CSV for test.csv."""
    pipeline = load_pipeline(model_path)
    logger.info("Loading test data (path=%s)...", test_path or "default")
    test_df = load_test(test_path)

    logger.info("Running inference on %d test rows...", len(test_df))
    submission_df = predict_probabilities(pipeline, test_df)

    if output_path is None:
        model_name = Path(model_path).stem
        out_dir = Path("outputs/submissions")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_out_path = out_dir / f"{model_name}_submission.csv"
    else:
        final_out_path = Path(output_path)
        final_out_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df.to_csv(final_out_path, index=False)
    logger.info("Submission successfully written to %s", final_out_path)
    return final_out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fraud probabilities using a trained ML pipeline.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to serialized ML pipeline (.joblib).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Optional path to test.csv. Uses datasets/raw/test.csv if not provided.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output submission CSV.",
    )
    args = parser.parse_args()

    generate_submission(
        model_path=args.model_path,
        test_path=args.test_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
