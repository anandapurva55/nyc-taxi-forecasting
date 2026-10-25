"""Optuna-based hyperparameter tuning for the tree models.

Each estimator is tuned by fitting on the training split and scoring on the
chronological validation split (minimising RMSE). Using the held-out validation
split - rather than random CV - preserves the temporal, leakage-safe evaluation
created by the data pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna
import pandas as pd

from src.models.common import fit_and_score

logger = logging.getLogger(__name__)

# Quieten Optuna's per-trial logging; the orchestrator logs summaries instead.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, estimator: str) -> dict[str, Any]:
    """Suggest a hyperparameter set for the given estimator."""
    if estimator == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if estimator == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(f"Unsupported estimator: {estimator!r}")


def tune_estimator(
    estimator: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int = 25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run an Optuna study for one estimator.

    Returns a dict with ``best_params``, ``best_value`` (validation RMSE) and the
    underlying ``study``.
    """

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, estimator)
        _, metrics = fit_and_score(
            estimator, params, X_train, y_train, X_val, y_val, random_state=random_state
        )
        return metrics["rmse"]

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Tuned %s: best val RMSE=%.4f over %d trials",
        estimator,
        study.best_value,
        n_trials,
    )
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
