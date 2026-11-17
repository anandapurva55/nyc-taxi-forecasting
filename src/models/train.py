"""Training orchestration.

For a given problem (demand or fare) this:

1. Loads the chronological splits.
2. Tunes each tree estimator (XGBoost, LightGBM) with Optuna on a holdout carved
   from the END of the training split, so the validation split is never used
   for tuning.
3. Fits each tuned candidate on the full training split and scores it on the
   validation split. Logs params/metrics/model to MLflow (one run per estimator).
4. Selects the best estimator by the configured validation metric, with a paired
   bootstrap interval showing whether the winner is clearly better.
5. Refits the winner on training plus validation data and evaluates it once on
   the held-out test split, with a block-bootstrap confidence interval.
6. Computes SHAP explanations and writes a model card (logged as MLflow artifacts).
   For fares it also learns a booking-time duration estimate.
7. Registers the final model and points the ``champion`` alias at it.

Training/evaluation logic lives in ``common``/``tuning`` so it can be unit-tested
without MLflow; this module only adds logging, selection and registration.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from src.config import Config, get_config
from src.features.duration import DurationLookup
from src.models import common, model_card, shap_utils, tuning, uncertainty
from src.utils.mlflow_utils import configure_mlflow, start_run

logger = logging.getLogger(__name__)

# Metrics where a larger value is better (others: smaller is better).
_HIGHER_IS_BETTER = {"r2"}

CHAMPION_ALIAS = "champion"
DURATION_LOOKUP_ARTIFACT = "duration_lookup.json"


def _is_better(metric: str, candidate: float, incumbent: float) -> bool:
    if metric in _HIGHER_IS_BETTER:
        return candidate > incumbent
    return candidate < incumbent


def train_problem(
    problem: str,
    config: Config | None = None,
    *,
    n_trials: int | None = None,
    register: bool = True,
) -> dict:
    """Train, tune, evaluate, explain and register models for ``problem``.

    Returns a summary dict describing each estimator's metrics and the winner.
    """
    if problem not in common.PROBLEMS:
        raise ValueError(f"Unknown problem: {problem!r} (expected 'demand' or 'fare')")

    config = config or get_config()
    model_cfg = config.models
    random_state = int(model_cfg.get("random_state", 42))
    n_trials = n_trials if n_trials is not None else int(model_cfg.get("n_trials", 25))
    estimators = list(model_cfg.get("estimators", ["xgboost", "lightgbm"]))
    holdout_frac = float(model_cfg.get("tuning_holdout_frac", 0.15))
    n_boot = int(model_cfg.get("bootstrap_samples", 500))
    metric = common.selection_metric(config)

    splits = common.load_splits(config, problem)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    X_fit, y_fit, X_tune, y_tune = common.chronological_holdout(X_train, y_train, holdout_frac)
    logger.info(
        "[%s] train=%d (tune on last %d) val=%d test=%d, tuning %s with %d trials each",
        problem, len(X_train), len(X_tune), len(X_val), len(X_test), estimators, n_trials,
    )

    results: list[dict] = []
    for estimator in estimators:
        tuned = tuning.tune_estimator(
            estimator, X_fit, y_fit, X_tune, y_tune,
            n_trials=n_trials, random_state=random_state,
        )
        best_params = tuned["best_params"]

        # Fit on the full training split; validation is used only for selection.
        model, val_metrics = common.fit_and_score(
            estimator, best_params, X_train, y_train, X_val, y_val,
            random_state=random_state,
        )
        val_pred = model.predict(X_val)

        with start_run(problem, run_name=f"{problem}-{estimator}", config=config) as run:
            mlflow.log_param("estimator", estimator)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("tuning_holdout_frac", holdout_frac)
            mlflow.log_params(best_params)
            mlflow.log_metric("tune_rmse", tuned["best_value"])
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(model, artifact_path="model")
            run_id = run.info.run_id

        logger.info("[%s] %s: val_%s=%.4f", problem, estimator, metric, val_metrics[metric])
        results.append(
            {
                "estimator": estimator,
                "params": best_params,
                "tune_rmse": tuned["best_value"],
                "val_metrics": val_metrics,
                "val_pred": val_pred,
                "run_id": run_id,
            }
        )

    # Select the winner by the validation selection metric.
    best = results[0]
    for cand in results[1:]:
        if _is_better(metric, cand["val_metrics"][metric], best["val_metrics"][metric]):
            best = cand
    logger.info("[%s] best estimator: %s", problem, best["estimator"])

    # How sure is the choice? Paired block bootstrap of the RMSE gap on validation.
    val_blocks = common.hour_blocks(config, problem, "val")
    selection = None
    others = [r for r in results if r is not best]
    if others:
        observed, low, high = uncertainty.paired_difference_ci(
            y_val.to_numpy(), best["val_pred"], others[0]["val_pred"], val_blocks,
            common.rmse, n_boot=n_boot, seed=random_state,
        )
        selection = {
            "winner": best["estimator"],
            "runner_up": others[0]["estimator"],
            "val_rmse_difference": observed,
            "ci95": [low, high],
            "clear_winner": high < 0,
        }

    # Final model: refit the winner on train + validation, evaluate once on test.
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    final_model = common.make_estimator(best["estimator"], best["params"], random_state=random_state)
    final_model.fit(X_full, y_full)
    test_pred = final_model.predict(X_test)
    test_metrics = common.regression_metrics(y_test.to_numpy(), test_pred)
    test_blocks = common.hour_blocks(config, problem, "test")
    rmse_low, rmse_high = uncertainty.block_bootstrap_ci(
        y_test.to_numpy(), test_pred, test_blocks, common.rmse, n_boot=n_boot, seed=random_state
    )
    test_metrics_ci = {"rmse": [rmse_low, rmse_high]}

    # Explainability + model card on the final model.
    artifacts_root = Path(config.models.get("artifacts_dir", "artifacts"))
    out_dir = (artifacts_root / problem).resolve()
    shap_out = shap_utils.save_shap_plots(
        final_model, X_full, out_dir, prefix=problem, random_state=random_state
    )
    registry_name = common.registry_model_name(problem, config)
    card = model_card.build_model_card(
        problem,
        best["estimator"],
        best["params"],
        test_metrics,
        best["val_metrics"],
        shap_out["importance"],
        data_year=int(config.data.get("tlc_year")) if config.data.get("tlc_year") else None,
        data_month=int(config.data.get("tlc_month")) if config.data.get("tlc_month") else None,
        n_train=len(X_full),
        registry_name=registry_name,
    )
    card_path = model_card.save_model_card(card, out_dir, prefix=problem)

    configure_mlflow(config)
    with start_run(problem, run_name=f"{problem}-{best['estimator']}-final", config=config) as run:
        mlflow.set_tag("best_model", "true")
        mlflow.set_tag("trained_on", "train+val")
        mlflow.log_param("estimator", best["estimator"])
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(best["params"])
        mlflow.log_metrics({f"val_{k}": v for k, v in best["val_metrics"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({"test_rmse_ci_low": rmse_low, "test_rmse_ci_high": rmse_high})
        mlflow.sklearn.log_model(final_model, artifact_path="model")
        mlflow.log_artifact(str(shap_out["summary"]), artifact_path="explainability")
        mlflow.log_artifact(str(shap_out["bar"]), artifact_path="explainability")
        mlflow.log_artifact(str(card_path), artifact_path="model_card")
        if problem == "fare":
            full_frame = pd.concat(
                [common.load_split_frame(config, problem, s) for s in ("train", "val")],
                ignore_index=True,
            )
            lookup = DurationLookup.fit(full_frame)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / DURATION_LOOKUP_ARTIFACT
                path.write_text(json.dumps(lookup.to_dict(), indent=2))
                mlflow.log_artifact(str(path))
        final_run_id = run.info.run_id

    registered = None
    if register:
        result = mlflow.register_model(f"runs:/{final_run_id}/model", registry_name)
        mlflow.MlflowClient().set_registered_model_alias(registry_name, CHAMPION_ALIAS, result.version)
        registered = {"name": registry_name, "version": int(result.version), "alias": CHAMPION_ALIAS}
        logger.info(
            "Registered %s version %s from run %s as @%s",
            registry_name, result.version, final_run_id, CHAMPION_ALIAS,
        )

    return {
        "problem": problem,
        "selection_metric": metric,
        "best_estimator": best["estimator"],
        "selection": selection,
        "results": [
            {
                "estimator": r["estimator"],
                "tune_rmse": r["tune_rmse"],
                "val_metrics": r["val_metrics"],
                "run_id": r["run_id"],
            }
            for r in results
        ],
        "final": {
            "trained_on": "train+val",
            "rows": len(X_full),
            "test_metrics": test_metrics,
            "test_metrics_ci95": test_metrics_ci,
            "run_id": final_run_id,
        },
        "registered": registered,
        "artifacts_dir": str(out_dir),
    }


def train_all(
    config: Config | None = None,
    *,
    problems: list[str] | None = None,
    n_trials: int | None = None,
    register: bool = True,
) -> dict[str, dict]:
    """Train every requested problem and return per-problem summaries."""
    problems = problems or list(common.PROBLEMS.keys())
    return {
        p: train_problem(p, config, n_trials=n_trials, register=register)
        for p in problems
    }
