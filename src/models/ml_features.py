"""Feature engineering transformer for Machine Learning models."""

from __future__ import annotations

from typing import Any, Final, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..data.loading import BINARY_FEATURES, NUMERIC_FEATURES

ENGINEERED_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "risk_flags_count",
    "amount_to_balance_ratio",
    "amount_by_category_zscore",
    "velocity_burst_rate",
    "cvv_retry_x_merchant_risk",
    "impossible_travel_speed",
    "time_sin",
    "time_cos",
    "channel_device_freq",
)

ALL_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    *NUMERIC_FEATURES,
    *ENGINEERED_NUMERIC_FEATURES,
)


def get_numeric_features(X: pd.DataFrame | np.ndarray) -> list[str]:
    """Dynamically return all numeric features (raw + engineered) present in X."""
    if isinstance(X, pd.DataFrame):
        return [c for c in ALL_NUMERIC_FEATURES if c in X.columns]
    return list(ALL_NUMERIC_FEATURES)


class MLFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer computing 9 high-value engineered numerical features for fraud detection.

    Guarantees 100% leak-free operation:
    - fit(X_train): Computes summary statistics (mean/std amount by category, pair frequencies,
      and global fallbacks) strictly on the training partition.
    - transform(X): Applies vectorised formulas, maps saved stats, safely handles zero-division
      and unseen categories, ensuring finite non-null outputs.
    """

    def __init__(self) -> None:
        self.category_amount_mean_: dict[str, float] = {}
        self.category_amount_std_: dict[str, float] = {}
        self.global_amount_mean_: float = 0.0
        self.global_amount_std_: float = 1.0
        self.channel_device_freqs_: dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> MLFeatureEngineer:
        """Fit statistical aggregations strictly on the provided training partition."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Global amount statistics for fallback
        if "amount_usd" in X.columns:
            self.global_amount_mean_ = float(X["amount_usd"].mean())
            std_val = float(X["amount_usd"].std())
            self.global_amount_std_ = std_val if np.isfinite(std_val) and std_val > 1e-5 else 1.0
        else:
            self.global_amount_mean_ = 0.0
            self.global_amount_std_ = 1.0

        # 2. Category amount mean and std aggregations
        if "merchant_category" in X.columns and "amount_usd" in X.columns:
            stats = X.groupby("merchant_category")["amount_usd"].agg(["mean", "std"])
            self.category_amount_mean_ = stats["mean"].to_dict()
            self.category_amount_std_ = (
                stats["std"].fillna(self.global_amount_std_).to_dict()
            )
        else:
            self.category_amount_mean_ = {}
            self.category_amount_std_ = {}

        # 3. Channel + device_type frequency table
        if "channel" in X.columns and "device_type" in X.columns:
            pairs = X["channel"].astype(str) + "_" + X["device_type"].astype(str)
            pair_counts = pairs.value_counts(normalize=True)
            self.channel_device_freqs_ = pair_counts.to_dict()
        else:
            self.channel_device_freqs_ = {}

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features and augment DataFrame with 9 new engineered numerical features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        # 1. risk_flags_count: sum of active boolean risk flags
        existing_binary = [c for c in BINARY_FEATURES if c in X_out.columns]
        if existing_binary:
            X_out["risk_flags_count"] = (
                X_out[existing_binary].astype(bool).astype(float).sum(axis=1)
            )
        else:
            X_out["risk_flags_count"] = 0.0

        # 2. amount_to_balance_ratio: ratio of transaction amount to account balance
        if "amount_usd" in X_out.columns and "account_balance_usd" in X_out.columns:
            safe_balance = (
                np.maximum(X_out["account_balance_usd"].to_numpy(dtype=float), 0.0) + 1.0
            )
            X_out["amount_to_balance_ratio"] = (
                X_out["amount_usd"].to_numpy(dtype=float) / safe_balance
            )
        else:
            X_out["amount_to_balance_ratio"] = 0.0

        # 3. amount_by_category_zscore: standardized amount relative to merchant category
        if "amount_usd" in X_out.columns and "merchant_category" in X_out.columns:
            cats = X_out["merchant_category"].astype(str)
            cat_mean = cats.map(self.category_amount_mean_).fillna(self.global_amount_mean_).to_numpy(dtype=float)
            cat_std = cats.map(self.category_amount_std_).fillna(self.global_amount_std_).to_numpy(dtype=float)
            safe_std = np.where(cat_std < 1e-5, self.global_amount_std_, cat_std)
            X_out["amount_by_category_zscore"] = (
                X_out["amount_usd"].to_numpy(dtype=float) - cat_mean
            ) / (safe_std + 1e-5)
        else:
            X_out["amount_by_category_zscore"] = 0.0

        # 4. velocity_burst_rate: velocity interaction with transaction count and latency
        if (
            "velocity_score" in X_out.columns
            and "txn_count_last_24h" in X_out.columns
            and "hours_since_last_txn" in X_out.columns
        ):
            safe_hours = (
                np.maximum(X_out["hours_since_last_txn"].to_numpy(dtype=float), 0.0) + 0.1
            )
            X_out["velocity_burst_rate"] = (
                X_out["velocity_score"].to_numpy(dtype=float)
                * X_out["txn_count_last_24h"].to_numpy(dtype=float)
            ) / safe_hours
        else:
            X_out["velocity_burst_rate"] = 0.0

        # 5. cvv_retry_x_merchant_risk: interaction of retry attempts and merchant risk
        if "cvv_retry_count" in X_out.columns and "merchant_risk_score" in X_out.columns:
            X_out["cvv_retry_x_merchant_risk"] = (
                X_out["cvv_retry_count"].to_numpy(dtype=float)
                * X_out["merchant_risk_score"].to_numpy(dtype=float)
            )
        else:
            X_out["cvv_retry_x_merchant_risk"] = 0.0

        # 6. impossible_travel_speed: distance from home divided by elapsed hours
        if "distance_from_home_km" in X_out.columns and "hours_since_last_txn" in X_out.columns:
            safe_hours = (
                np.maximum(X_out["hours_since_last_txn"].to_numpy(dtype=float), 0.0) + 0.1
            )
            X_out["impossible_travel_speed"] = (
                X_out["distance_from_home_km"].to_numpy(dtype=float) / safe_hours
            )
        else:
            X_out["impossible_travel_speed"] = 0.0

        # 7 & 8. Cyclical time encoding (time_sin & time_cos)
        if "time_of_day_hour" in X_out.columns:
            hour_arr = X_out["time_of_day_hour"].to_numpy(dtype=float)
            X_out["time_sin"] = np.sin(2.0 * np.pi * hour_arr / 24.0)
            X_out["time_cos"] = np.cos(2.0 * np.pi * hour_arr / 24.0)
        else:
            X_out["time_sin"] = 0.0
            X_out["time_cos"] = 0.0

        # 9. channel_device_freq: frequency encoding for channel + device combination
        if "channel" in X_out.columns and "device_type" in X_out.columns:
            pairs = X_out["channel"].astype(str) + "_" + X_out["device_type"].astype(str)
            X_out["channel_device_freq"] = pairs.map(self.channel_device_freqs_).fillna(0.0).to_numpy(dtype=float)
        else:
            X_out["channel_device_freq"] = 0.0

        # Safeguard: Clean any NaN or inf values in newly created features
        for col in ENGINEERED_NUMERIC_FEATURES:
            if col in X_out.columns:
                X_out[col] = np.nan_to_num(
                    X_out[col].to_numpy(dtype=float),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        return X_out
