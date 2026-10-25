"""Shared model-development utilities.

Centralises regression metrics, loading of the split parquets into
``(X, y)`` matrices, and an estimator factory for the two supported tree models
(XGBoost and LightGBM). Keeping these here means the training orchestration,
tuning and tests all share one source of truth and stay consistent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import Config, get_config
from src.features import demand_features, fare_features

logger = logging.getLogger(__name__)

# Per-problem metadata: target column, the module exposing ``feature_columns``
# and the config key holding the MLflow registry model name.
PROBLEMS: dict[str, dict[str, Any]] = {
    "demand": {
        "target": demand_features.TARGET,
        "feature_columns": demand_features.feature_columns,
        "registry_key": "registry_demand",
        "registry_default": "taxi-demand-forecaster",
        "file_prefix": "demand",
    },
    "fare": {
        "target": fare_features.TARGET,
        "feature_columns": fare_features.feature_columns,
        "registry_key": "registry_fare",
        "registry_default": "taxi-fare-predictor",
        "file_prefix": "fare",
    },
}


# --- Metrics ---------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Mean absolute percentage error (%), ignoring near-zero targets.

    Records with ``|y_true| < eps`` are excluded to avoid division blow-ups
    (e.g. zero-demand hours); returns ``nan`` if no records remain.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.abs(y_true) >= eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the standard regression metric suite as a dict."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


# --- Data loading ----------------------------------------------------------
def _split_path(config: Config, prefix: str, split: str) -> Path:
    return config.features_dir / f"{prefix}_{split}.parquet"


def split_to_xy(
    df: pd.DataFrame, problem: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature DataFrame into model inputs ``X`` and target ``y``."""
    meta = PROBLEMS[problem]
    target = meta["target"]
    feature_cols: list[str] = meta["feature_columns"](df)
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y


def load_split(config: Config, problem: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a single (train/val/test) split as ``(X, y)`` for a problem."""
    meta = PROBLEMS[problem]
    path = _split_path(config, meta["file_prefix"], split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split parquet: {path}. Run the pipeline first "
            f"(python -m scripts.run_pipeline)."
        )
    df = pd.read_parquet(path)
    return split_to_xy(df, problem)


def load_splits(
    config: Config, problem: str
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Load train/val/test splits for a problem keyed by split name."""
    return {split: load_split(config, problem, split) for split in ("train", "val", "test")}


# --- Estimator factory -----------------------------------------------------
def make_estimator(
    name: str, params: dict[str, Any] | None = None, *, random_state: int = 42
):
    """Instantiate an XGBoost or LightGBM regressor with sensible defaults."""
    params = dict(params or {})
    if name == "xgboost":
        defaults = {
            "n_estimators": 300,
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        defaults.update(params)
        return XGBRegressor(**defaults)
    if name == "lightgbm":
        defaults = {
            "n_estimators": 300,
            "objective": "regression",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": -1,
        }
        defaults.update(params)
        return LGBMRegressor(**defaults)
    raise ValueError(f"Unsupported estimator: {name!r} (expected 'xgboost' or 'lightgbm')")


def fit_and_score(
    name: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    random_state: int = 42,
):
    """Fit an estimator and return ``(fitted_model, metrics_on_eval)``."""
    model = make_estimator(name, params, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    return model, regression_metrics(y_eval.to_numpy(), preds)


def registry_model_name(problem: str, config: Config | None = None) -> str:
    """Return the configured MLflow registry name for a problem."""
    config = config or get_config()
    meta = PROBLEMS[problem]
    return config.models.get(meta["registry_key"], meta["registry_default"])


def selection_metric(config: Config | None = None) -> str:
    config = config or get_config()
    return config.models.get("selection_metric", "rmse")
