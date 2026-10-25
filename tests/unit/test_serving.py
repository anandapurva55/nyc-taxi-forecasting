"""Tests for serving feature builders, predictions and batch logic.

Fully offline: a small sklearn model is trained on synthetic-derived features and
injected, so no MLflow registry is required.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.data.aggregation import aggregate_hourly_demand
from src.features.demand_features import build_demand_features
from src.models import common
from src.serving import batch, features, predict, registry, service


@pytest.fixture()
def demand_history(clean_trips_df):
    return aggregate_hourly_demand(clean_trips_df)


@pytest.fixture()
def demand_model(clean_trips_df):
    demand = aggregate_hourly_demand(clean_trips_df)
    feats = build_demand_features(demand)
    X, y = common.split_to_xy(feats, "demand")
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture()
def fare_model(clean_trips_df):
    from src.features.fare_features import build_fare_features

    feats = build_fare_features(clean_trips_df)
    X, y = common.split_to_xy(feats, "fare")
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)


# --- Feature builders ------------------------------------------------------
def test_build_demand_row_shape_and_columns(demand_history):
    zone = int(demand_history["PULocationID"].iloc[0])
    target = pd.Timestamp(demand_history["pickup_hour"].max())
    row = features.build_demand_row(demand_history, zone, target)


    # Column set/order matches what the model was trained on.
    assert list(row.columns)  # non-empty
    assert "lag_1" in row.columns and "rolling_mean_3" in row.columns
    assert len(row) == 1
    assert row.isna().sum().sum() == 0
    # lag_1 equals the previous hour's pickup count from history.
    prev = demand_history[
        (demand_history["PULocationID"] == zone)
        & (demand_history["pickup_hour"] == target - pd.Timedelta(hours=1))
    ]["pickup_count"]
    if not prev.empty:
        assert row["lag_1"].iloc[0] == prev.iloc[0]


def test_build_fare_row_estimates_duration():
    row_no_dur = features.build_fare_row(
        trip_distance=4.0,
        passenger_count=2,
        pu_location_id=1,
        do_location_id=2,
        pickup_datetime=pd.Timestamp("2024-01-15T10:00:00"),
    )
    assert row_no_dur["trip_duration_min"].iloc[0] == pytest.approx(4.0 * features.DEFAULT_MIN_PER_MILE)
    assert "fare_amount" not in row_no_dur.columns


def test_estimate_duration_uses_provided_value():
    assert features.estimate_duration(5.0, 12.5) == 12.5
    assert features.estimate_duration(5.0, None) == pytest.approx(15.0)


# --- Predictions -----------------------------------------------------------
def test_predict_fare_returns_float(fare_model):
    pred = predict.predict_fare(
        fare_model,
        trip_distance=3.0,
        passenger_count=1,
        pu_location_id=1,
        do_location_id=2,
        pickup_datetime=pd.Timestamp("2024-01-15T18:00:00"),
    )
    assert isinstance(pred, float) and math.isfinite(pred)


def test_predict_demand_non_negative(demand_model, demand_history):
    zone = int(demand_history["PULocationID"].iloc[0])
    target = pd.Timestamp(demand_history["pickup_hour"].max())
    pred = predict.predict_demand(demand_model, demand_history, zone=zone, target_hour=target)
    assert pred >= 0.0


def test_model_service_caches_demand_prediction_by_zone_and_hour(demand_history, monkeypatch):
    from src.serving.service import ModelService

    calls = []

    def fake_predict(*args, zone, target_hour, **kwargs):
        calls.append((zone, target_hour))
        return 12.5

    monkeypatch.setattr(predict, "predict_demand", fake_predict)
    service = ModelService(None, None, demand_history)
    target = pd.Timestamp("2024-01-15T18:00:00")

    assert service.predict_demand(zone=1, target_hour=target) == 12.5
    assert service.predict_demand(zone=1, target_hour=target + pd.Timedelta(minutes=30)) == 12.5
    assert service.predict_demand(zone=1, target_hour=target + pd.Timedelta(hours=1)) == 12.5
    assert calls == [(1, target), (1, target + pd.Timedelta(hours=1))]


def test_registry_uri_and_load_model(monkeypatch):
    from src.config import Config

    config = Config(
        raw={
            "models": {"registry_fare": "fare-v2"},
            "serving": {"model_stage": "7"},
            "mlflow": {"tracking_uri": "file:./test-runs"},
        }
    )
    configured = []
    loaded = []
    sentinel = object()
    monkeypatch.setattr(registry, "configure_mlflow", configured.append)
    monkeypatch.setattr(
        registry.mlflow,
        "sklearn",
        SimpleNamespace(load_model=lambda uri: loaded.append(uri) or sentinel),
    )

    assert registry.model_uri("fare", config) == "models:/fare-v2/7"
    assert registry.load_model("fare", config) is sentinel
    assert configured == [config]
    assert loaded == ["models:/fare-v2/7"]


def test_registry_rebases_host_artifact_path_for_container(monkeypatch, tmp_path):
    from mlflow.exceptions import MlflowException

    from src.config import Config

    config = Config(
        raw={
            "models": {"registry_fare": "fare-v2"},
            "serving": {"model_stage": "latest"},
            "mlflow": {"tracking_uri": "sqlite:////app/mlflow.db"},
        }
    )
    container_runs = tmp_path / "mlruns"
    model_path = container_runs / "2" / "run-id" / "artifacts" / "model"
    model_path.mkdir(parents=True)
    loaded = []
    sentinel = object()

    def fake_load_model(uri):
        loaded.append(uri)
        if uri.startswith("models:/"):
            raise MlflowException("Host artifact path is unavailable")
        return sentinel

    client = SimpleNamespace(
        search_model_versions=lambda query: [
            SimpleNamespace(
                name="fare-v2",
                version="7",
                source="file:///C:/taxi-forecasting/mlruns/2/run-id/artifacts/model",
            )
        ]
        if query == "name='fare-v2'"
        else []
    )
    monkeypatch.setattr(registry, "_CONTAINER_MLRUNS", container_runs)
    monkeypatch.setattr(registry, "configure_mlflow", lambda _config: None)
    monkeypatch.setattr(registry.mlflow, "MlflowClient", lambda: client)
    monkeypatch.setattr(
        registry.mlflow, "sklearn", SimpleNamespace(load_model=fake_load_model)
    )

    assert registry.load_model("fare", config) is sentinel
    assert loaded == ["models:/fare-v2/latest", model_path.as_uri()]


def test_load_demand_history_normalizes_timestamp(tmp_path):
    from src.config import Config

    path = tmp_path / "history.parquet"
    pd.DataFrame(
        {"PULocationID": [1], "pickup_hour": ["2024-01-01T08:00:00"], "pickup_count": [3]}
    ).to_parquet(path, index=False)
    config = Config(raw={"serving": {"demand_history_path": str(path)}})

    history = service.load_demand_history(config)

    assert pd.api.types.is_datetime64_any_dtype(history["pickup_hour"])


def test_load_demand_history_reports_missing_path(tmp_path):
    from src.config import Config

    config = Config(raw={"serving": {"demand_history_path": str(tmp_path / "missing.parquet")}})

    with pytest.raises(FileNotFoundError, match="Run the pipeline"):
        service.load_demand_history(config)


def test_model_service_from_registry_and_lazy_provider(monkeypatch, demand_history):
    from src.config import Config

    config = Config(raw={})
    models = {"demand": object(), "fare": object()}
    monkeypatch.setattr(registry, "load_model", lambda problem, config: models[problem])
    monkeypatch.setattr(service, "load_demand_history", lambda config: demand_history)

    loaded = service.ModelService.from_registry(config)
    assert loaded.demand_model is models["demand"]
    assert loaded.fare_model is models["fare"]
    assert loaded.known_zones()

    service.set_service(None)
    monkeypatch.setattr(service.ModelService, "from_registry", lambda: loaded)
    assert service.get_service() is loaded
    assert service.get_service() is loaded
    service.set_service(None)


def test_model_service_caches_explainers(monkeypatch, demand_history):
    from src.responsible import explain

    built = []
    monkeypatch.setattr(explain, "build_explainer", lambda model: built.append(model) or object())
    demand_model = object()
    fare_model = object()
    model_service = service.ModelService(demand_model, fare_model, demand_history)

    assert model_service.fare_explainer() is model_service.fare_explainer()
    assert model_service.demand_explainer() is model_service.demand_explainer()
    assert built == [fare_model, demand_model]


# --- Batch -----------------------------------------------------------------
def test_forecast_demand_recursive(demand_model, demand_history):
    from src.serving.service import ModelService

    svc = ModelService(demand_model, None, demand_history)
    zones = svc.known_zones()[:2]
    result = batch.forecast_demand(svc, horizon_hours=3, zones=zones)

    assert len(result) == 3 * len(zones)
    assert (result["predicted_pickups"] >= 0).all()
    # Hours are contiguous and start right after the latest history hour.
    expected_start = svc.latest_history_hour() + pd.Timedelta(hours=1)
    assert result["pickup_hour"].min() == expected_start


def test_write_forecasts_to_sqlite(demand_model, demand_history, tmp_path):
    import sqlite3

    from src.config import load_config
    from src.serving.service import ModelService

    config = load_config()
    config.raw["serving"]["batch_db_path"] = str(tmp_path / "preds.db")

    svc = ModelService(demand_model, None, demand_history, config)
    result = batch.forecast_demand(svc, horizon_hours=2, zones=svc.known_zones()[:1], config=config)
    db_path = batch.write_forecasts(result, config, model_version="test")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM demand_forecasts").fetchone()[0]
    assert rows == len(result)
