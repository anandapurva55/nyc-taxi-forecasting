"""Evaluation beyond a single test score: baselines, horizons, imputation and backtests.

- **Baselines.** A model only shows skill if it beats simple rules. Demand is
  compared with "same as last hour" (persistence), "same hour yesterday" and
  "same hour last week" (seasonal naive). Fare is compared with linear
  regressions on distance and duration, and on distance alone.
- **Horizons.** Scoring one step ahead with true lags does not measure a
  multi-hour forecast, where the model's own predictions become its lags. The
  real recursive forecaster is started at many origins and scored at every hour
  ahead, next to the naive rules. This is what sets how far ahead the API uses
  the model (``serving.model_horizon_hours``) before switching to the same hour
  last week.
- **Booking-time fares.** Callers rarely know the trip duration, so fares are
  also scored without it: with estimated durations and with the booking-time
  model trained without duration.
- **Rolling-origin backtest.** The winning configuration is refit at several
  cut-off dates and scored on the following days, giving a spread rather than
  one number from one window.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import demand_features, fare_features
from src.features.duration import DurationLookup
from src.models import common, uncertainty
from src.responsible.fairness import wape

logger = logging.getLogger(__name__)

DEMAND_BASELINES = {
    "persistence (last hour)": "lag_1",
    "same hour yesterday": "lag_24",
    "same hour last week": "lag_168",
}


def score(y_true, y_pred, blocks=None, n_boot: int = 500) -> dict:
    """RMSE, MAE, WAPE and R2, plus a block-bootstrap 95% interval for RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "rmse": common.rmse(y_true, y_pred),
        "mae": common.mae(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": common.r2(y_true, y_pred),
        "n": int(len(y_true)),
    }
    if blocks is not None:
        low, high = uncertainty.block_bootstrap_ci(y_true, y_pred, blocks, common.rmse, n_boot=n_boot)
        out["rmse_ci95"] = [low, high]
    return out


# --- Baselines ---------------------------------------------------------------
def demand_baseline_scores(test: pd.DataFrame, model_pred: np.ndarray, n_boot: int = 500) -> dict:
    """Score the model and each naive rule on the same one-step-ahead test rows."""
    y = test[demand_features.TARGET].to_numpy(dtype=float)
    blocks = pd.to_datetime(test[demand_features.TIME_COL]).dt.floor("h").to_numpy()
    results = {"model": score(y, model_pred, blocks, n_boot)}
    for name, col in DEMAND_BASELINES.items():
        results[name] = score(y, test[col].to_numpy(dtype=float), blocks, n_boot)
    return results


def fare_baseline_scores(
    train: pd.DataFrame, test: pd.DataFrame, model_pred: np.ndarray, n_boot: int = 500
) -> dict:
    """Score the model against linear regressions fitted on the same training rows."""
    y = test[fare_features.TARGET].to_numpy(dtype=float)
    blocks = pd.to_datetime(test[fare_features.TIME_COL]).dt.floor("h").to_numpy()
    results = {"model": score(y, model_pred, blocks, n_boot)}
    for name, cols in (
        ("linear: distance + duration", ["trip_distance", "trip_duration_min"]),
        ("linear: distance only", ["trip_distance"]),
    ):
        lr = LinearRegression().fit(train[cols], train[fare_features.TARGET])
        results[name] = score(y, lr.predict(test[cols]), blocks, n_boot)
    return results


# --- Multi-hour demand forecasts ----------------------------------------------
def forecast_origins(test_start: pd.Timestamp, last_hour: pd.Timestamp, horizon: int, step: int):
    """Forecast start hours inside the test window that leave room for the full horizon."""
    origins = []
    origin = pd.Timestamp(test_start)
    while origin + pd.Timedelta(hours=horizon - 1) <= last_hour:
        origins.append(origin)
        origin += pd.Timedelta(hours=step)
    return origins


def horizon_scores(
    demand_model,
    history: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
    horizon: int,
    config,
    zones: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Recursive forecasts from each origin, joined to actuals and naive baselines.

    For each origin the forecaster only sees history before that hour, exactly
    as the API would. Returns one row per (origin, zone, hours ahead) with the
    model forecast, the actual, persistence from the origin, and the seasonal
    naive values (same hour yesterday is only known in advance up to 24 hours).
    """
    from src.serving import batch
    from src.serving.service import ModelService

    hist = history.copy()
    hist["pickup_hour"] = pd.to_datetime(hist["pickup_hour"])
    actual = hist.set_index(["PULocationID", "pickup_hour"])["pickup_count"]
    rows = []
    for origin in origins:
        past = hist[hist["pickup_hour"] < origin]
        service = ModelService(demand_model, None, past, config)
        # model_hours=horizon scores the model on its own at every hour ahead.
        fc = batch.forecast_demand(
            service, horizon_hours=horizon, start_hour=origin,
            zones=list(zones) if zones else None, model_hours=horizon,
        )
        fc["origin"] = origin
        fc["hours_ahead"] = ((fc["pickup_hour"] - origin) / pd.Timedelta(hours=1)).astype(int) + 1
        last_seen = past[past["pickup_hour"] == origin - pd.Timedelta(hours=1)].set_index("PULocationID")["pickup_count"]
        fc["persistence"] = fc["PULocationID"].map(last_seen).fillna(0).to_numpy()
        for name, lag in (("same_hour_yesterday", 24), ("same_hour_last_week", 168)):
            keys = list(zip(fc["PULocationID"], fc["pickup_hour"] - pd.Timedelta(hours=lag), strict=True))
            fc[name] = actual.reindex(keys).fillna(0).to_numpy()
        keys = list(zip(fc["PULocationID"], fc["pickup_hour"], strict=True))
        fc["actual"] = actual.reindex(keys).to_numpy()
        rows.append(fc)
        logger.info("Horizon evaluation: origin %s done (%d forecasts)", origin, len(fc))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["actual"])
    # Yesterday's value is not yet observed when forecasting more than 24 hours ahead.
    out.loc[out["hours_ahead"] > 24, "same_hour_yesterday"] = np.nan
    return out


def summarise_horizons(
    frame: pd.DataFrame,
    model_hours: int,
    report_at: Sequence[int] = (1, 2, 3, 6, 12, 24),
    n_boot: int = 300,
) -> dict:
    """RMSE by hours ahead for the model, the served forecast and each baseline.

    ``served`` is what the API returns: the model up to ``model_hours`` ahead and
    the same hour last week after that. ``model_minus_last_week_ci95`` is a paired
    bootstrap interval resampled by forecast origin; below zero means the model
    is more accurate than the weekly rule.
    """
    frame = frame.copy()
    frame["served"] = np.where(
        frame["hours_ahead"] <= model_hours, frame["predicted_pickups"], frame["same_hour_last_week"]
    )

    def row(part: pd.DataFrame) -> dict:
        y = part["actual"].to_numpy(dtype=float)
        out = {"n": int(len(part))}
        for name in ("predicted_pickups", "served", "persistence", "same_hour_yesterday", "same_hour_last_week"):
            vals = part[name]
            key = "model" if name == "predicted_pickups" else name
            out[key] = common.rmse(y[vals.notna()], vals.dropna().to_numpy()) if vals.notna().any() else None
        diff, low, high = uncertainty.paired_difference_ci(
            y, part["predicted_pickups"].to_numpy(dtype=float), part["same_hour_last_week"].to_numpy(dtype=float),
            pd.to_datetime(part["origin"]).to_numpy(), common.rmse, n_boot=n_boot,
        )
        out["model_minus_last_week_ci95"] = [diff, low, high]
        return out

    by_hour = [{"hours_ahead": h, **row(frame[frame["hours_ahead"] == h])} for h in report_at
               if (frame["hours_ahead"] == h).any()]
    return {"model_hours": model_hours, "by_hours_ahead": by_hour, "all_hours": row(frame)}


# --- Booking-time fares ---------------------------------------------------------
def fare_duration_scores(
    fare_model,
    test: pd.DataFrame,
    lookup: DurationLookup | None,
    fixed_min_per_mile: float,
    n_boot: int = 500,
    booking_model=None,
) -> dict:
    """Fare accuracy with the true duration versus the ways of quoting without it."""
    X = test[fare_features.feature_columns(test)].copy()
    y = test[fare_features.TARGET].to_numpy(dtype=float)
    blocks = pd.to_datetime(test[fare_features.TIME_COL]).dt.floor("h").to_numpy()
    results = {"true duration": score(y, fare_model.predict(X), blocks, n_boot)}
    fixed = X.copy()
    fixed["trip_duration_min"] = np.maximum(1.0, fixed["trip_distance"] * fixed_min_per_mile)
    results["fixed 3 min per mile"] = score(y, fare_model.predict(fixed), blocks, n_boot)
    if lookup is not None:
        learned = X.copy()
        learned["trip_duration_min"] = [
            lookup.estimate(d, h) for d, h in zip(learned["trip_distance"], learned["hour"], strict=True)
        ]
        results["learned estimate"] = score(y, fare_model.predict(learned), blocks, n_boot)
    if booking_model is not None:
        booking = X.drop(columns=list(fare_features.BOOKING_EXCLUDED))
        results["booking-time model"] = score(y, booking_model.predict(booking), blocks, n_boot)
    return results


# --- Rolling-origin backtest ------------------------------------------------------
def rolling_backtest(
    frame: pd.DataFrame,
    problem: str,
    estimator: str,
    params: dict,
    cutoffs: Sequence[pd.Timestamp],
    test_days: int,
    baseline,
    random_state: int = 42,
) -> list[dict]:
    """Refit at each cut-off and score the next ``test_days`` days against a baseline.

    ``baseline`` maps (train_frame, test_frame) to baseline predictions for the
    test rows, so model and baseline are always scored on identical rows.
    """
    time_col = common.PROBLEMS[problem]["time_col"]
    times = pd.to_datetime(frame[time_col])
    folds = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        train = frame[times < cutoff]
        test = frame[(times >= cutoff) & (times < cutoff + pd.Timedelta(days=test_days))]
        if train.empty or test.empty:
            continue
        X_train, y_train = common.split_to_xy(train, problem)
        X_test, y_test = common.split_to_xy(test, problem)
        model = common.make_estimator(estimator, params, random_state=random_state).fit(X_train, y_train)
        y = y_test.to_numpy(dtype=float)
        folds.append(
            {
                "cutoff": str(cutoff),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "model_rmse": common.rmse(y, model.predict(X_test)),
                "baseline_rmse": common.rmse(y, np.asarray(baseline(train, test), dtype=float)),
            }
        )
        logger.info("Backtest %s fold %s: model RMSE %.3f", problem, cutoff, folds[-1]["model_rmse"])
    return folds


def fold_summary(folds: list[dict]) -> dict:
    model = np.array([f["model_rmse"] for f in folds])
    base = np.array([f["baseline_rmse"] for f in folds])
    return {
        "n_folds": len(folds),
        "model_rmse_mean": float(model.mean()) if len(model) else None,
        "model_rmse_std": float(model.std(ddof=1)) if len(model) > 1 else None,
        "baseline_rmse_mean": float(base.mean()) if len(base) else None,
        "model_beats_baseline_every_fold": bool(len(folds) and np.all(model < base)),
    }
