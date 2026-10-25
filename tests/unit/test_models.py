"""Tests for model utilities (metrics, loading, tuning).

These run fully offline against synthetic-derived features and never touch
MLflow or the Model Registry - training/metric logic is deliberately separated
from logging/registration in :mod:`src.models`.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.data.aggregation import aggregate_hourly_demand
from src.features.demand_features import build_demand_features
from src.features.fare_features import build_fare_features
from src.models import common, model_card, shap_utils, train, tuning


# --- Metrics ---------------------------------------------------------------
def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = common.regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["mape"] == 0.0
    assert m["r2"] == 1.0


def test_metrics_known_values():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([3.0, 4.0, 5.0])  # errors: +1, 0, -1
    assert common.mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert common.rmse(y_true, y_pred) == pytest.approx(math.sqrt(2 / 3))


def test_mape_ignores_near_zero_targets():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    # Only the 100 -> 110 record counts: 10% error.
    assert common.mape(y_true, y_pred) == pytest.approx(10.0)


# --- Data shaping ----------------------------------------------------------
def test_split_to_xy_excludes_target(clean_trips_df):
    feats = build_fare_features(clean_trips_df)
    X, y = common.split_to_xy(feats, "fare")
    assert "fare_amount" not in X.columns
    assert y.name == "fare_amount"
    assert len(X) == len(y)


# --- End-to-end training (tiny, offline) -----------------------------------
def _demand_xy(clean_trips_df):
    demand = aggregate_hourly_demand(clean_trips_df)
    feats = build_demand_features(demand)
    return common.split_to_xy(feats, "demand")


@pytest.mark.parametrize("estimator", ["xgboost", "lightgbm"])
def test_fit_and_score_produces_finite_metrics(clean_trips_df, estimator):
    X, y = _demand_xy(clean_trips_df)
    half = len(X) // 2
    model, metrics = common.fit_and_score(
        estimator,
        {"n_estimators": 20},
        X.iloc[:half],
        y.iloc[:half],
        X.iloc[half:],
        y.iloc[half:],
    )
    assert model is not None
    for key in ("rmse", "mae", "mape", "r2"):
        assert math.isfinite(metrics[key])


def test_tune_estimator_single_trial(clean_trips_df):
    X, y = _demand_xy(clean_trips_df)
    half = len(X) // 2
    result = tuning.tune_estimator(
        "lightgbm",
        X.iloc[:half],
        y.iloc[:half],
        X.iloc[half:],
        y.iloc[half:],
        n_trials=1,
    )
    assert "best_params" in result
    assert math.isfinite(result["best_value"])


def test_build_model_card_contains_training_evidence():
    card = model_card.build_model_card(
        "demand",
        "lightgbm",
        {"n_estimators": 20},
        {"rmse": 2.0, "mae": 1.0, "mape": 5.0, "r2": 0.9},
        {"rmse": 2.5, "mae": 1.2, "mape": 6.0, "r2": 0.85},
        pd.Series({"lag_1": 4.2, "hour": 1.5}),
        data_year=2024,
        data_month=1,
        n_train=1234,
        registry_name="taxi-demand",
    )

    assert card.startswith("# Model Card - taxi-demand")
    assert "**Training data window:** 2024-01" in card
    assert "**Training rows:** 1,234" in card
    assert "| RMSE | 2.0000 | 2.5000 |" in card
    assert "1. `lag_1` - 4.2000" in card
    assert '"n_estimators": 20' in card
    assert "Only Yellow Taxi pickups are modelled" in card


def test_save_model_card_writes_requested_prefix(tmp_path):
    path = model_card.save_model_card("# Fare model", tmp_path, prefix="fare")

    assert path == tmp_path / "fare_model_card.md"
    assert path.read_text(encoding="utf-8") == "# Fare model"


@pytest.mark.parametrize(
    ("metric", "candidate", "incumbent", "expected"),
    [("r2", 0.9, 0.8, True), ("r2", 0.7, 0.8, False), ("rmse", 2.0, 3.0, True)],
)
def test_is_better_respects_metric_direction(metric, candidate, incumbent, expected):
    assert train._is_better(metric, candidate, incumbent) is expected


def test_train_all_forwards_options(monkeypatch):
    calls = []

    def fake_train_problem(problem, config, *, n_trials, register):
        calls.append((problem, config, n_trials, register))
        return {"problem": problem}

    monkeypatch.setattr(train, "train_problem", fake_train_problem)
    config = object()

    result = train.train_all(config, problems=["fare", "demand"], n_trials=3, register=False)

    assert result == {"fare": {"problem": "fare"}, "demand": {"problem": "demand"}}
    assert calls == [("fare", config, 3, False), ("demand", config, 3, False)]


def test_train_problem_selects_winner_logs_artifacts_and_registers(monkeypatch, tmp_path):
    from src.config import Config

    config = Config(
        raw={
            "data": {"tlc_year": 2024, "tlc_month": 1},
            "models": {
                "estimators": ["first", "second"],
                "selection_metric": "rmse",
                "artifacts_dir": str(tmp_path),
                "registry_demand": "demand-model",
            },
        }
    )
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([2.0, 4.0], name="pickup_count")
    monkeypatch.setattr(
        common,
        "load_splits",
        lambda config, problem: {split: (X, y) for split in ("train", "val", "test")},
    )
    monkeypatch.setattr(
        tuning,
        "tune_estimator",
        lambda estimator, *args, **kwargs: {"best_params": {"name": estimator}},
    )

    class FakeModel:
        def predict(self, features):
            return np.zeros(len(features))

    validation_rmse = {"first": 3.0, "second": 2.0}
    monkeypatch.setattr(
        common,
        "fit_and_score",
        lambda estimator, *args, **kwargs: (
            FakeModel(),
            {"rmse": validation_rmse[estimator], "mae": 1.0, "mape": 2.0, "r2": 0.5},
        ),
    )
    monkeypatch.setattr(
        common,
        "regression_metrics",
        lambda *args: {"rmse": 2.5, "mae": 1.5, "mape": 3.0, "r2": 0.4},
    )

    @contextmanager
    def fake_start_run(problem=None, run_name=None, config=None, run_id=None):
        del problem, config
        yield SimpleNamespace(info=SimpleNamespace(run_id=run_id or f"run-{run_name}"))

    logged_artifacts = []
    monkeypatch.setattr(train, "start_run", fake_start_run)
    monkeypatch.setattr(train.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(train.mlflow, "log_param", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "set_tag", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train.mlflow,
        "log_artifact",
        lambda path, artifact_path=None: logged_artifacts.append((path, artifact_path)),
    )
    monkeypatch.setattr(
        train.mlflow,
        "sklearn",
        SimpleNamespace(log_model=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        train.mlflow,
        "register_model",
        lambda uri, name: SimpleNamespace(version="7", uri=uri, name=name),
    )
    monkeypatch.setattr(train, "configure_mlflow", lambda config: None)
    monkeypatch.setattr(
        shap_utils,
        "save_shap_plots",
        lambda *args, **kwargs: {
            "summary": tmp_path / "summary.png",
            "bar": tmp_path / "bar.png",
            "importance": pd.Series({"feature": 1.0}),
        },
    )
    monkeypatch.setattr(model_card, "build_model_card", lambda *args, **kwargs: "card")
    monkeypatch.setattr(
        model_card,
        "save_model_card",
        lambda *args, **kwargs: tmp_path / "demand_model_card.md",
    )

    result = train.train_problem("demand", config, n_trials=2)

    assert result["best_estimator"] == "second"
    assert result["registered"] == {"name": "demand-model", "version": "7"}
    assert result["results"][1]["run_id"] == "run-demand-second"
    assert len(logged_artifacts) == 3


def test_train_problem_rejects_unknown_problem():
    with pytest.raises(ValueError, match="Unknown problem"):
        train.train_problem("unknown")


def test_save_shap_plots_writes_artifacts_and_ranked_importance(tmp_path):
    features = pd.DataFrame(
        {
            "strong": np.arange(20, dtype=float),
            "weak": np.tile([0.0, 1.0], 10),
        }
    )
    target = features["strong"] * 3.0
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(features, target)

    result = shap_utils.save_shap_plots(
        model, features, tmp_path, prefix="tiny", max_samples=10, random_state=0
    )

    assert result["summary"] == tmp_path / "tiny_shap_summary.png"
    assert result["bar"] == tmp_path / "tiny_shap_bar.png"
    assert result["summary"].stat().st_size > 0
    assert result["bar"].stat().st_size > 0
    assert list(result["importance"].index) == ["strong", "weak"]
