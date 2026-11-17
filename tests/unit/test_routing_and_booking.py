"""Unit tests for horizon routing of demand forecasts and the booking-time fare model."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.data.aggregation import aggregate_hourly_demand
from src.features.demand_features import build_demand_features
from src.features.fare_features import (
    BOOKING_EXCLUDED,
    booking_feature_columns,
    build_fare_features,
)
from src.models import common
from src.monitoring.live_performance import join_demand_outcomes
from src.serving import batch
from src.serving.service import ModelService

FARE_KWARGS = {
    "trip_distance": 3.4,
    "passenger_count": 1,
    "pu_location_id": 1,
    "do_location_id": 2,
    "pickup_datetime": pd.Timestamp("2024-02-27T18:30:00"),
}


@pytest.fixture()
def demand_history(clean_trips_df):
    return aggregate_hourly_demand(clean_trips_df)


@pytest.fixture()
def demand_model(demand_history):
    X, y = common.split_to_xy(build_demand_features(demand_history), "demand")
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)


@pytest.fixture()
def fare_models(clean_trips_df):
    feats = build_fare_features(clean_trips_df)
    X, y = common.split_to_xy(feats, "fare")
    Xb, yb = common.split_to_xy(feats, "fare_booking")
    return (
        DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y),
        DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xb, yb),
    )


# --- Demand routing ----------------------------------------------------------
def test_seasonal_naive_uses_latest_observed_week():
    last = pd.Timestamp("2024-02-29 23:00")
    observed = {pd.Timestamp("2024-02-24 10:00"): 7, pd.Timestamp("2024-02-17 10:00"): 3}
    assert batch.seasonal_naive(observed, pd.Timestamp("2024-03-02 10:00"), last) == 7.0
    # Eight days ahead: last week's hour is not observed yet, so go back another week.
    assert batch.seasonal_naive(observed, pd.Timestamp("2024-03-09 10:00"), last) == 7.0
    assert batch.seasonal_naive(observed, pd.Timestamp("2024-03-02 11:00"), last) == 0.0


def test_forecast_uses_model_then_last_week(demand_model, demand_history):
    svc = ModelService(demand_model, None, demand_history)
    zone = svc.known_zones()[0]
    result = batch.forecast_demand(svc, horizon_hours=5, zones=[zone], model_hours=2)

    assert list(result["method"]) == [batch.MODEL] * 2 + [batch.SEASONAL_NAIVE] * 3
    zone_hist = demand_history[demand_history["PULocationID"] == zone]
    observed = dict(zip(pd.to_datetime(zone_hist["pickup_hour"]), zone_hist["pickup_count"], strict=True))
    last = svc.latest_history_hour()
    for row in result[result["method"] == batch.SEASONAL_NAIVE].itertuples():
        assert row.predicted_pickups == batch.seasonal_naive(observed, row.pickup_hour, last)


def test_model_hours_override_scores_model_everywhere(demand_model, demand_history):
    svc = ModelService(demand_model, None, demand_history)
    result = batch.forecast_demand(svc, horizon_hours=4, zones=svc.known_zones()[:1], model_hours=4)
    assert set(result["method"]) == {batch.MODEL}


def test_single_hour_prediction_follows_the_same_routing(demand_model, demand_history):
    svc = ModelService(demand_model, None, demand_history)
    zone = svc.known_zones()[0]
    last = svc.latest_history_hour()
    forecast = batch.forecast_demand(svc, horizon_hours=6, zones=[zone]).set_index("pickup_hour")
    for ahead in (1, 2, 6):
        target = last + pd.Timedelta(hours=ahead)
        assert svc.predict_demand(zone=zone, target_hour=target) == pytest.approx(
            forecast.loc[target, "predicted_pickups"]
        )
    assert svc.demand_method(last + pd.Timedelta(hours=2)) == batch.MODEL
    assert svc.demand_method(last + pd.Timedelta(hours=3)) == batch.SEASONAL_NAIVE


def test_explain_demand_rejects_hours_with_unobserved_inputs(demand_model, demand_history):
    svc = ModelService(demand_model, None, demand_history)
    with pytest.raises(ValueError, match="one hour past"):
        svc.explain_demand(zone=svc.known_zones()[0], target_hour=svc.latest_history_hour() + pd.Timedelta(hours=2))


def test_write_forecasts_adds_method_to_older_tables(demand_model, demand_history, tmp_path):
    from src.config import load_config

    db = tmp_path / "predictions.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE demand_forecasts (PULocationID INTEGER NOT NULL, pickup_hour TEXT NOT NULL, "
            "predicted_pickups REAL NOT NULL, generated_at TEXT NOT NULL, model_version TEXT NOT NULL)"
        )
    config = load_config()
    config.raw["serving"]["batch_db_path"] = str(db)
    svc = ModelService(demand_model, None, demand_history, config)
    rows = batch.forecast_demand(svc, horizon_hours=3, zones=svc.known_zones()[:1])
    batch.write_forecasts(rows, config, model_version="taxi-demand-forecaster/2")
    with sqlite3.connect(db) as conn:
        methods = [r[0] for r in conn.execute("SELECT method FROM demand_forecasts ORDER BY pickup_hour")]
    assert methods == [batch.MODEL, batch.MODEL, batch.SEASONAL_NAIVE]


def test_live_performance_scores_only_model_served_hours():
    captured = pd.DataFrame(
        {
            "zone": [132, 230],
            "target_hour": ["2024-02-29T20:00:00", "2024-02-29T20:00:00"],
            "prediction": [90.0, 10.0],
            "method": [batch.MODEL, batch.SEASONAL_NAIVE],
        }
    )
    actuals = pd.DataFrame(
        {"PULocationID": [132, 230], "pickup_hour": pd.to_datetime(["2024-02-29 20:00"] * 2), "pickup_count": [100, 40]}
    )
    joined = join_demand_outcomes(captured, actuals)
    assert list(joined["zone"]) == [132]


# --- Booking-time fare ---------------------------------------------------------
def test_booking_features_drop_only_the_duration(clean_trips_df):
    feats = build_fare_features(clean_trips_df)
    cols = booking_feature_columns(feats)
    assert set(common.PROBLEMS["fare"]["feature_columns"](feats)) - set(cols) == set(BOOKING_EXCLUDED)


def test_fare_routes_to_booking_model_only_without_duration(fare_models, demand_history):
    fare_model, booking_model = fare_models
    svc = ModelService(None, fare_model, demand_history, fare_booking_model=booking_model)

    assert svc.fare_problem(None) == "fare_booking"
    assert svc.fare_problem(12.0) == "fare"
    assert svc.fare_without_duration() == "booking-time model"

    from src.serving import predict

    expected_booking = predict.predict_fare(booking_model, booking=True, **FARE_KWARGS)
    expected_full = predict.predict_fare(fare_model, trip_duration_min=12.0, **FARE_KWARGS)
    assert svc.predict_fare(**FARE_KWARGS) == expected_booking
    assert svc.predict_fare(trip_duration_min=12.0, **FARE_KWARGS) == expected_full


def test_without_booking_model_fare_falls_back_to_duration_estimate(fare_models, demand_history):
    fare_model, _ = fare_models
    svc = ModelService(None, fare_model, demand_history)
    assert svc.fare_problem(None) == "fare"
    assert svc.fare_without_duration() == "fixed duration estimate"
