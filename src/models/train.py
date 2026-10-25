"""Training orchestration.

For a given problem (demand or fare) this:

1. Loads the chronological splits.
2. Tunes each tree estimator (XGBoost, LightGBM) with Optuna on the validation
   split, then fits the best configuration.
3. Logs params/metrics/model to MLflow (one run per estimator).
4. Selects the best estimator by the configured validation metric.
5. Evaluates the winner on the held-out test split.
6. Computes SHAP explanations and writes a model card (logged as MLflow artifacts).
7. Registers the winning model in the MLflow Model Registry.

Training/evaluation logic lives in ``common``/``tuning`` so it can be unit-tested
without MLflow; this module only adds logging, selection and registration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow

from src.config import Config, get_config
from src.models import common, model_card, shap_utils, tuning
from src.utils.mlflow_utils import configure_mlflow, start_run

logger = logging.getLogger(__name__)

# Metrics where a larger value is better (others: smaller is better).
_HIGHER_IS_BETTER = {"r2"}


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
    metric = common.selection_metric(config)

    splits = common.load_splits(config, problem)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    logger.info(
        "[%s] train=%d val=%d test=%d, tuning %s with %d trials each",
        problem,
        len(X_train),
        len(X_val),
        len(X_test),
        estimators,
        n_trials,
    )

    results: list[dict] = []
    for estimator in estimators:
        tuned = tuning.tune_estimator(
            estimator,
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials,
            random_state=random_state,
        )
        best_params = tuned["best_params"]

        # Refit on train with the best params and evaluate on val + test.
        model, val_metrics = common.fit_and_score(
            estimator, best_params, X_train, y_train, X_val, y_val,
            random_state=random_state,
        )
        test_metrics = common.regression_metrics(
            y_test.to_numpy(), model.predict(X_test)
        )

        with start_run(problem, run_name=f"{problem}-{estimator}", config=config) as run:
            mlflow.log_param("estimator", estimator)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_params(best_params)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.sklearn.log_model(model, artifact_path="model")
            run_id = run.info.run_id

        logger.info(
            "[%s] %s: val_%s=%.4f test_%s=%.4f",
            problem, estimator, metric, val_metrics[metric], metric, test_metrics[metric],
        )
        results.append(
            {
                "estimator": estimator,
                "params": best_params,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "model": model,
                "run_id": run_id,
            }
        )

    # Select the winner by the validation selection metric.
    best = results[0]
    for cand in results[1:]:
        if _is_better(metric, cand["val_metrics"][metric], best["val_metrics"][metric]):
            best = cand
    logger.info("[%s] best estimator: %s", problem, best["estimator"])

    # Explainability + model card + registration on the winning run.
    artifacts_root = Path(config.models.get("artifacts_dir", "artifacts"))
    out_dir = (artifacts_root / problem).resolve()
    shap_out = shap_utils.save_shap_plots(
        best["model"], X_train, out_dir, prefix=problem, random_state=random_state
    )
    registry_name = common.registry_model_name(problem, config)
    card = model_card.build_model_card(
        problem,
        best["estimator"],
        best["params"],
        best["test_metrics"],
        best["val_metrics"],
        shap_out["importance"],
        data_year=int(config.data.get("tlc_year")) if config.data.get("tlc_year") else None,
        data_month=int(config.data.get("tlc_month")) if config.data.get("tlc_month") else None,
        n_train=len(X_train),
        registry_name=registry_name,
    )
    card_path = model_card.save_model_card(card, out_dir, prefix=problem)

    # Re-open the winning run to attach artifacts and register the model.
    model_uri = f"runs:/{best['run_id']}/model"
    configure_mlflow(config)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("best_model", "true")
        mlflow.log_artifact(str(shap_out["summary"]), artifact_path="explainability")
        mlflow.log_artifact(str(shap_out["bar"]), artifact_path="explainability")
        mlflow.log_artifact(str(card_path), artifact_path="model_card")

    registered = None
    if register:
        result = mlflow.register_model(model_uri, registry_name)
        registered = {"name": registry_name, "version": result.version}
        logger.info(
            "Registered %s version %s from run %s",
            registry_name, result.version, best["run_id"],
        )

    return {
        "problem": problem,
        "selection_metric": metric,
        "best_estimator": best["estimator"],
        "results": [
            {
                "estimator": r["estimator"],
                "val_metrics": r["val_metrics"],
                "test_metrics": r["test_metrics"],
                "run_id": r["run_id"],
            }
            for r in results
        ],
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
