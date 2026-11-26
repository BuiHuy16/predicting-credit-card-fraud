from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loading import (
    BINARY_FEATURES,
    NUMERIC_FEATURES,
    load_train,
)
from src.data.preprocessing import make_preprocessor
from src.data.splitting import load_split, validation_split_id
from src.models.mlp import FraudMLP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dl.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def torch_version() -> str:
    try:
        return str(torch.__version__)
    except AttributeError:
        return str(torch.version.__version__)


def dense_float32(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Preprocessed features contain NaN or Inf.")
    return np.ascontiguousarray(array)


def make_criterion(
    targets: np.ndarray,
    *,
    class_weighting: bool,
    device: torch.device,
) -> tuple[nn.BCEWithLogitsLoss, float | None]:
    if not class_weighting:
        return nn.BCEWithLogitsLoss(), None

    labels = np.asarray(targets).reshape(-1)
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(np.count_nonzero(labels == 0))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("pos_weight requires both target classes in the training partition.")

    pos_weight = negative_count / positive_count
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor), pos_weight


def make_split_preprocessor(excluded: tuple[str, ...]):
    import src.data.preprocessing as preprocessing_module

    original_numeric = preprocessing_module.NUMERIC_FEATURES
    original_binary = preprocessing_module.BINARY_FEATURES
    preprocessing_module.NUMERIC_FEATURES = tuple(
        column for column in NUMERIC_FEATURES if column not in excluded
    )
    preprocessing_module.BINARY_FEATURES = tuple(
        column for column in BINARY_FEATURES if column not in excluded
    )
    try:
        return make_preprocessor(scale_numeric=True, sparse_threshold=0.0)
    finally:
        preprocessing_module.NUMERIC_FEATURES = original_numeric
        preprocessing_module.BINARY_FEATURES = original_binary


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            logits = model(features)
            loss = criterion(logits, target)
            losses.append(float(loss.item()) * features.shape[0])
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            targets.append(target.cpu().numpy().reshape(-1))
    probability_array = np.concatenate(probabilities)
    target_array = np.concatenate(targets)
    loss = float(sum(losses) / len(target_array))
    if not np.isfinite(probability_array).all():
        raise ValueError("Validation probabilities contain NaN or Inf.")
    return loss, probability_array, target_array


def smoke_test(config: dict[str, Any]) -> None:
    set_seed(int(config["seed"]))
    train = load_train()
    split = load_split(train)
    excluded = tuple(config["preprocessing"]["temporarily_excluded_features"])

    X_train = split.X_train.iloc[:512].drop(columns=list(excluded), errors="raise").copy()
    X_validation = (
        split.X_validation.iloc[:128].drop(columns=list(excluded), errors="raise").copy()
    )
    y_train = split.y_train.iloc[:512].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_validation = split.y_validation.iloc[:128].to_numpy(dtype=np.float32).reshape(-1, 1)

    preprocessor = make_split_preprocessor(excluded)
    train_array = dense_float32(preprocessor.fit_transform(X_train))
    validation_array = dense_float32(preprocessor.transform(X_validation))

    device = torch.device("cpu")
    model = FraudMLP(
        input_dim=train_array.shape[1],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    batch_features = torch.from_numpy(train_array[:16])
    logits = model(batch_features)
    if logits.shape != (16, 1) or not torch.isfinite(logits).all():
        raise RuntimeError("Smoke forward pass failed shape or finite checks.")
    criterion, pos_weight = make_criterion(
        split.y_train.to_numpy(dtype=np.float32),
        class_weighting=bool(config["training"].get("class_weighting", False)),
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    target = torch.from_numpy(y_train[:16])
    optimizer.zero_grad()
    loss = criterion(logits, target)
    loss.backward()
    if model.network[0].weight.grad is None:
        raise RuntimeError("Smoke backward pass did not produce gradients.")
    optimizer.step()

    train_dataset = TensorDataset(torch.from_numpy(train_array), torch.from_numpy(y_train))
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_array), torch.from_numpy(y_validation)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    _, probabilities, targets = evaluate(model, validation_loader, criterion, device)
    if probabilities.shape[0] != 128 or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError("Smoke validation probability contract failed.")
    print(
        json.dumps(
            {
                "smoke": "pass",
                "input_dim": train_array.shape[1],
                "train_shape": list(train_array.shape),
                "validation_shape": list(validation_array.shape),
                "probability_count": int(probabilities.shape[0]),
                "target_classes": sorted(int(value) for value in np.unique(targets)),
                "pos_weight": pos_weight,
            },
            ensure_ascii=False,
        )
    )


def train_baseline(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    train = load_train()
    split = load_split(train)
    artifact_ids = pd.read_csv(
        PROJECT_ROOT / "experiments" / "validation" / "validation_ids_seed42.csv"
    )["id"].tolist()
    split_id = validation_split_id(artifact_ids)
    if split_id != config["split_id"]:
        raise ValueError(f"Unexpected split_id: {split_id}")
    if not set(split.train_ids).isdisjoint(set(split.validation_ids)):
        raise ValueError("Train and validation IDs overlap.")
    if set(split.train_ids) | set(split.validation_ids) != set(train["id"]):
        raise ValueError("Train/validation partitions do not cover the training data exactly.")

    excluded = tuple(config["preprocessing"]["temporarily_excluded_features"])
    X_train = split.X_train.drop(columns=list(excluded), errors="raise").copy()
    X_validation = split.X_validation.drop(columns=list(excluded), errors="raise").copy()
    feature_order = list(X_train.columns)

    preprocessor = make_split_preprocessor(excluded)
    train_array = dense_float32(preprocessor.fit_transform(X_train))
    validation_array = dense_float32(preprocessor.transform(X_validation))
    y_train = split.y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_validation = split.y_validation.to_numpy(dtype=np.float32).reshape(-1, 1)

    if y_train.shape[0] != train_array.shape[0]:
        raise ValueError("Training feature/target row counts differ.")
    if y_validation.shape[0] != validation_array.shape[0]:
        raise ValueError("Validation feature/target row counts differ.")

    input_dim = int(train_array.shape[1])
    device = torch.device("cpu")
    model = FraudMLP(
        input_dim=input_dim,
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    training = config["training"]
    criterion, pos_weight = make_criterion(
        y_train,
        class_weighting=bool(training.get("class_weighting", False)),
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_array), torch.from_numpy(y_train)),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_array), torch.from_numpy(y_validation)),
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )

    run_id = str(config["run_id"])
    run_dir = PROJECT_ROOT / "experiments" / "dl" / run_id
    model_dir = PROJECT_ROOT / "outputs" / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    actual_config = copy.deepcopy(config)
    actual_config["runtime"] = {
        "device": str(device),
        "python": platform.python_version(),
        "pytorch": torch_version(),
        "input_dim": input_dim,
        "parameter_count": parameter_count,
        "feature_order": feature_order,
        "class_weighting": bool(training.get("class_weighting", False)),
        "pos_weight": pos_weight,
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(actual_config, handle, sort_keys=False)

    log_path = run_dir / "train_log.csv"
    log_fields = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_roc_auc",
        "validation_pr_auc",
        "learning_rate",
        "epoch_seconds",
    ]
    log_rows: list[dict[str, Any]] = []
    best_roc_auc = -float("inf")
    best_stopping_roc_auc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    min_delta = float(training["early_stopping_min_delta"])
    patience = int(training["early_stopping_patience"])
    checkpoint_path = model_dir / f"{run_id}.pt"
    preprocessor_path = run_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Training loss became NaN/Inf at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * features.shape[0]
            train_rows += features.shape[0]

        train_loss = train_loss_sum / train_rows
        validation_loss, probabilities, targets = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        roc_auc = float(roc_auc_score(targets, probabilities))
        pr_auc = float(average_precision_score(targets, probabilities))
        epoch_seconds = time.perf_counter() - epoch_started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_roc_auc": roc_auc,
            "validation_pr_auc": pr_auc,
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        log_rows.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} roc_auc={roc_auc:.6f} "
            f"pr_auc={pr_auc:.6f}",
            flush=True,
        )

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "architecture": {
                        "input_dim": input_dim,
                        "hidden_dims": list(config["model"]["hidden_dims"]),
                        "dropout": float(config["model"]["dropout"]),
                    },
                    "run_id": run_id,
                    "model": "FraudMLP",
                    "split_id": split_id,
                    "training_seed": seed,
                    "split_seed": int(config["seed_split"]),
                    "best_epoch": best_epoch,
                    "best_validation_roc_auc": best_roc_auc,
                    "feature_order": feature_order,
                    "temporarily_excluded_features": list(excluded),
                    "class_weighting": bool(training.get("class_weighting", False)),
                    "pos_weight": pos_weight,
                    "preprocessor_path": str(preprocessor_path),
                    "preprocessor_checksum": None,
                    "pytorch_version": torch_version(),
                    "device": str(device),
                },
                checkpoint_path,
            )

        if roc_auc > best_stopping_roc_auc + min_delta:
            best_stopping_roc_auc = roc_auc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early_stopping at epoch={epoch}", flush=True)
                break

    training_seconds = time.perf_counter() - started
    pd.DataFrame(log_rows, columns=log_fields).to_csv(log_path, index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FraudMLP(
        input_dim=checkpoint["architecture"]["input_dim"],
        hidden_dims=tuple(checkpoint["architecture"]["hidden_dims"]),
        dropout=float(checkpoint["architecture"]["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss, probabilities, targets = evaluate(
        model,
        validation_loader,
        criterion,
        device,
    )
    roc_auc = float(roc_auc_score(targets, probabilities))
    pr_auc = float(average_precision_score(targets, probabilities))

    predictions = pd.DataFrame(
        {
            "id": split.validation_ids.to_numpy(),
            "y_true": targets.astype(np.int64),
            "is_fraud": probabilities.astype(np.float64),
            "run_id": run_id,
            "model": "FraudMLP",
            "split_id": split_id,
        }
    )
    predictions_path = run_dir / "validation_probabilities.csv"
    predictions.to_csv(predictions_path, index=False)

    metrics = {
        "run_id": run_id,
        "model": "FraudMLP",
        "split_id": split_id,
        "seed": seed,
        "seed_split": int(config["seed_split"]),
        "feature_order": feature_order,
        "temporarily_excluded_features": list(excluded),
        "class_weighting": bool(training.get("class_weighting", False)),
        "pos_weight": pos_weight,
        "preprocessing": "train-fit dense float32 preprocessor",
        "train_rows": int(train_array.shape[0]),
        "validation_rows": int(validation_array.shape[0]),
        "train_fraud_count": int(split.y_train.sum()),
        "validation_fraud_count": int(split.y_validation.sum()),
        "input_dim": input_dim,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "validation_roc_auc": roc_auc,
        "validation_pr_auc": pr_auc,
        "validation_loss": validation_loss,
        "train_seconds": training_seconds,
        "checkpoint_path": str(checkpoint_path),
        "preprocessor_path": str(preprocessor_path),
        "validation_predictions_path": str(predictions_path),
        "train_log_path": str(log_path),
        "config_path": str(run_dir / "config.yaml"),
        "device": str(device),
        "pytorch_version": torch_version(),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.smoke:
        smoke_test(config)
    else:
        train_baseline(config)


if __name__ == "__main__":
    main()
