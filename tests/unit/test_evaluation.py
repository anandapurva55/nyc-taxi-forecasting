"""Unit tests for the evaluation helpers (baselines, horizons, backtest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import evaluation


def test_forecast_origins_leave_room_for_the_horizon():
    origins = evaluation.forecast_origins(
        pd.Timestamp("2024-02-27 19:00"), pd.Timestamp("2024-02-29 23:00"), 24, 6
    )
    assert origins[0] == pd.Timestamp("2024-02-27 19:00")
    assert all(o + pd.Timedelta(hours=23) <= pd.Timestamp("2024-02-29 23:00") for o in origins)
    assert len(origins) == 5


def test_demand_baselines_score_the_naive_rules():
    hours = pd.date_range("2024-02-27", periods=48, freq="h")
    test = pd.DataFrame(
        {
            "pickup_hour": hours,
            "pickup_count": np.arange(48, dtype=float),
            "lag_1": np.arange(48, dtype=float) - 1,
            "lag_24": np.arange(48, dtype=float),
            "lag_168": np.arange(48, dtype=float) + 5,
        }
    )
    scores = evaluation.demand_baseline_scores(test, test["pickup_count"].to_numpy(), n_boot=50)
    assert scores["model"]["rmse"] == 0.0
    assert scores["same hour yesterday"]["rmse"] == 0.0
    assert scores["persistence (last hour)"]["rmse"] == 1.0
    assert scores["same hour last week"]["rmse"] == 5.0
    assert len(scores["model"]["rmse_ci95"]) == 2


def _horizon_frame():
    return pd.DataFrame(
        {
            "origin": pd.to_datetime(["2024-02-22 00:00", "2024-02-22 03:00", "2024-02-22 00:00"]),
            "hours_ahead": [1, 1, 30],
            "actual": [10.0, 20.0, 5.0],
            "predicted_pickups": [10.0, 18.0, 9.0],
            "persistence": [9.0, 19.0, 4.0],
            "same_hour_yesterday": [10.0, 20.0, np.nan],
            "same_hour_last_week": [11.0, 21.0, 6.0],
        }
    )


def test_summarise_horizons_hides_unknown_yesterday_beyond_24h():
    summary = evaluation.summarise_horizons(_horizon_frame(), model_hours=2, report_at=(1, 30), n_boot=20)
    rows = {r["hours_ahead"]: r for r in summary["by_hours_ahead"]}
    assert rows[1]["model"] == np.sqrt(2.0)
    assert rows[30]["same_hour_yesterday"] is None
    assert summary["all_hours"]["n"] == 3


def test_served_forecast_switches_to_last_week_beyond_model_hours():
    summary = evaluation.summarise_horizons(_horizon_frame(), model_hours=2, report_at=(1, 30), n_boot=20)
    rows = {r["hours_ahead"]: r for r in summary["by_hours_ahead"]}
    assert rows[1]["served"] == rows[1]["model"]
    # 30 hours ahead the API serves last week's count (error 1), not the model (error 4).
    assert rows[30]["served"] == 1.0 and rows[30]["model"] == 4.0


def test_rolling_backtest_scores_model_and_baseline_on_same_rows():
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-02-01", periods=24 * 20, freq="h")
    frame = pd.DataFrame(
        {
            "PULocationID": 1,
            "pickup_hour": times,
            "pickup_count": rng.poisson(20, len(times)).astype(float),
            "hour": times.hour,
        }
    )
    frame["lag_1"] = frame["pickup_count"].shift(1).fillna(20)
    folds = evaluation.rolling_backtest(
        frame, "demand", "lightgbm", {"n_estimators": 20}, [pd.Timestamp("2024-02-15")], 2,
        baseline=lambda train, test: np.full(len(test), train["pickup_count"].mean()),
    )
    assert len(folds) == 1 and folds[0]["test_rows"] == 48
    summary = evaluation.fold_summary(folds)
    assert summary["n_folds"] == 1 and summary["model_rmse_mean"] > 0
