"""Evaluate the registered champion models beyond a single test score.

Writes artifacts/evaluation/evaluation.json with:

- demand and fare baselines on the held-out test split, with 95% intervals;
- multi-hour demand accuracy from the real recursive forecaster, refit before
  2024-02-22 and started every few hours over the following week, next to the
  forecast the API actually serves (model, then same hour last week);
- fare accuracy without the trip duration, as at booking time;
- a rolling-origin backtest of the winning configuration for both models.

Usage::

    python -m scripts.evaluate_forecasts                 # full evaluation
    python -m scripts.evaluate_forecasts --quick         # daily origins, no backtest
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

from src.config import get_config
from src.features import demand_features, fare_features
from src.models import common, evaluation
from src.serving import batch, registry
from src.serving.features import DEFAULT_MIN_PER_MILE
from src.utils.mlflow_utils import configure_mlflow

logger = logging.getLogger(__name__)

# Multi-hour demand accuracy is measured on the week after this date, with the
# champion configuration refit on everything before it.
HORIZON_CUTOFF = pd.Timestamp("2024-02-22")


def _winner_config(problem: str, config) -> tuple[str, dict]:
    """Estimator name and tuned hyperparameters of the served champion."""
    info = registry.resolve_version(problem, config)
    params = mlflow.MlflowClient().get_run(info["run_id"]).data.params
    estimator = params.pop("estimator")
    params.pop("n_trials", None)
    params.pop("tuning_holdout_frac", None)
    typed = {}
    for key, value in params.items():
        try:
            typed[key] = int(value)
        except ValueError:
            try:
                typed[key] = float(value)
            except ValueError:
                typed[key] = value
    return estimator, typed


def run(quick: bool, origin_step: int, horizon: int) -> dict:
    config = get_config()
    configure_mlflow(config)
    demand_model = registry.load_model("demand", config)
    fare_model = registry.load_model("fare", config)
    versions = {p: registry.resolve_version(p, config) for p in ("demand", "fare")}
    n_boot = 200 if quick else int(config.models.get("bootstrap_samples", 500))

    # 1. Baselines on the held-out test split.
    demand_test = common.load_split_frame(config, "demand", "test")
    X_dt, _ = common.split_to_xy(demand_test, "demand")
    demand_baselines = evaluation.demand_baseline_scores(
        demand_test, demand_model.predict(X_dt).clip(min=0), n_boot
    )
    fare_test = common.load_split_frame(config, "fare", "test")
    fare_trainval = pd.concat(
        [common.load_split_frame(config, "fare", s) for s in ("train", "val")], ignore_index=True
    )
    X_ft, _ = common.split_to_xy(fare_test, "fare")
    fare_baselines = evaluation.fare_baseline_scores(
        fare_trainval, fare_test, fare_model.predict(X_ft), n_boot
    )

    # 2. Multi-hour demand accuracy over a full week of forecast origins.
    history = pd.read_parquet(config.processed_dir / "demand_hourly.parquet")
    history["pickup_hour"] = pd.to_datetime(history["pickup_hour"])
    demand_frame = pd.read_parquet(config.features_dir / "demand_features.parquet")
    estimator, params = _winner_config("demand", config)
    before = demand_frame[pd.to_datetime(demand_frame[demand_features.TIME_COL]) < HORIZON_CUTOFF]
    X_b, y_b = common.split_to_xy(before, "demand")
    refit = common.make_estimator(estimator, params, random_state=42).fit(X_b, y_b)
    step = 24 if quick else origin_step
    origins = evaluation.forecast_origins(HORIZON_CUTOFF, history["pickup_hour"].max(), horizon, step)
    frame = evaluation.horizon_scores(refit, history, origins, horizon, config)
    horizons = evaluation.summarise_horizons(frame, batch.model_horizon(config), n_boot=min(n_boot, 300))
    horizons["protocol"] = {
        "refit_before": str(HORIZON_CUTOFF),
        "origins": len(origins),
        "origin_step_hours": step,
        "first_origin": str(origins[0]),
        "last_origin": str(origins[-1]),
    }

    # 3. Fare accuracy with the duration estimated as at booking time.
    lookup = registry.load_duration_lookup(config)
    booking_model = registry.load_model("fare_booking", config)
    versions["fare_booking"] = registry.resolve_version("fare_booking", config)
    fare_duration = evaluation.fare_duration_scores(
        fare_model, fare_test, lookup, DEFAULT_MIN_PER_MILE, n_boot, booking_model=booking_model
    )

    result = {
        "models": versions,
        "demand": {"baselines": demand_baselines, "horizons": horizons},
        "fare": {"baselines": fare_baselines, "duration": fare_duration},
    }

    # 4. Rolling-origin backtest of the winning configurations.
    if not quick:
        cutoffs = [pd.Timestamp("2024-02-15"), pd.Timestamp("2024-02-19"), pd.Timestamp("2024-02-23")]
        folds = evaluation.rolling_backtest(
            demand_frame, "demand", estimator, params, cutoffs, 3,
            baseline=lambda train, test: test["lag_168"],
        )
        result["demand"]["backtest"] = {"baseline": "same hour last week", "folds": folds,
                                        **evaluation.fold_summary(folds)}

        fare_frame = pd.read_parquet(config.features_dir / "fare_features.parquet")
        estimator, params = _winner_config("fare", config)
        cols = ["trip_distance", "trip_duration_min"]

        def linear(train, test):
            from sklearn.linear_model import LinearRegression

            return LinearRegression().fit(train[cols], train[fare_features.TARGET]).predict(test[cols])

        folds = evaluation.rolling_backtest(
            fare_frame, "fare", estimator, params, cutoffs, 3, baseline=linear
        )
        result["fare"]["backtest"] = {"baseline": "linear: distance + duration", "folds": folds,
                                      **evaluation.fold_summary(folds)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="Daily origins, fewer resamples, no backtest.")
    parser.add_argument("--origin-step", type=int, default=3, help="Hours between forecast origins.")
    parser.add_argument("--horizon", type=int, default=24, help="Hours ahead to forecast.")
    parser.add_argument("--output", default="artifacts/evaluation/evaluation.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    result = run(args.quick, args.origin_step, args.horizon)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({
        "demand_baselines": {k: round(v["rmse"], 3) for k, v in result["demand"]["baselines"].items()},
        "demand_horizons": {
            str(r["hours_ahead"]): {k: round(r[k], 3) for k in ("model", "served", "same_hour_last_week")}
            for r in result["demand"]["horizons"]["by_hours_ahead"]
        },
        "fare_baselines": {k: round(v["rmse"], 3) for k, v in result["fare"]["baselines"].items()},
        "fare_duration": {k: round(v["rmse"], 3) for k, v in result["fare"]["duration"].items()},
    }, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
