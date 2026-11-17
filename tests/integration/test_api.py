"""Integration tests for the FastAPI prediction service.

Uses FastAPI's TestClient with a ModelService built from small in-process models,
so the API is exercised end-to-end without loading from the MLflow registry.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pandas as pd
import pytest

from src.serving.api import _capture_prediction


def test_capture_prediction_emits_structured_info_record():
    capture_logger = logging.getLogger("taxi.capture")
    capture_stream = StringIO()
    capture_handler = logging.StreamHandler(capture_stream)
    capture_handler.setFormatter(logging.Formatter("%(message)s"))
    capture_logger.addHandler(capture_handler)
    try:
        _capture_prediction("request-1", "fare", "2", {"trip_distance": 3.4}, 14.2)
    finally:
        capture_logger.removeHandler(capture_handler)

    payload = json.loads(capture_stream.getvalue())
    assert payload["event"] == "prediction"
    assert payload["request_id"] == "request-1"
    assert payload["problem"] == "fare"
    assert payload["model_version"] == "2"
    assert payload["features"] == {"trip_distance": 3.4}
    assert payload["prediction"] == 14.2
    assert capture_logger.propagate is False


def test_health_ok(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["demand_model_loaded"] is True
    assert body["fare_model_loaded"] is True
    assert body["zones"] > 0
    assert "X-Request-ID" in resp.headers


def test_predict_fare_endpoint(client):
    c, _ = client
    resp = c.post(
        "/predict/fare",
        json={
            "trip_distance": 3.4,
            "pu_location_id": 1,
            "do_location_id": 2,
            "passenger_count": 1,
            "pickup_datetime": "2024-01-15T18:30:00",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_fare"] >= 0
    assert body["duration_estimated"] is True
    assert body["request_id"]


def test_predict_demand_endpoint(client):
    c, demand = client
    zone = int(demand["PULocationID"].iloc[0])
    target = pd.Timestamp(demand["pickup_hour"].max())
    resp = c.post(
        "/predict/demand",
        json={"zone": zone, "target_hour": target.isoformat()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["zone"] == zone
    assert body["predicted_pickups"] >= 0


def test_predict_demand_unknown_zone_returns_400(client):
    c, _ = client
    resp = c.post(
        "/predict/demand",
        json={"zone": 99999, "target_hour": "2024-01-31T20:00:00"},
    )
    assert resp.status_code == 400


def test_predict_fare_validation_error(client):
    c, _ = client
    resp = c.post(
        "/predict/fare",
        json={
            "trip_distance": -1.0,  # invalid: must be > 0
            "pu_location_id": 1,
            "do_location_id": 2,
            "pickup_datetime": "2024-01-15T18:30:00",
        },
    )
    assert resp.status_code == 422


def test_request_id_is_propagated(client):
    c, _ = client
    resp = c.get("/health", headers={"X-Request-ID": "test-trace-123"})
    assert resp.headers["X-Request-ID"] == "test-trace-123"


def test_metrics_endpoint_exposes_prometheus(client):
    c, _ = client
    # Generate at least one request so counters are populated.
    c.get("/health")
    resp = c.get("/metrics")
    assert resp.status_code == 200
    # Prometheus exposition format includes HELP/TYPE comments.
    assert "# HELP" in resp.text
    assert "http_request" in resp.text


def test_explain_fare_endpoint(client):
    c, _ = client
    resp = c.post(
        "/explain/fare",
        json={
            "trip_distance": 3.4,
            "pu_location_id": 1,
            "do_location_id": 2,
            "passenger_count": 1,
            "pickup_datetime": "2024-01-15T18:30:00",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["problem"] == "fare"
    assert body["contributions"]
    # SHAP additivity: base + sum(contributions) ~= explained prediction.
    total = body["base_value"] + sum(c["value"] for c in body["contributions"])
    assert total == pytest.approx(body["prediction"], abs=1e-2)
    # Contributions are ranked by absolute impact.
    abs_vals = [abs(c["value"]) for c in body["contributions"]]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_explain_demand_endpoint(client):
    c, demand = client
    zone = int(demand["PULocationID"].iloc[0])
    target = pd.Timestamp(demand["pickup_hour"].max())
    resp = c.post(
        "/explain/demand",
        json={"zone": zone, "target_hour": target.isoformat()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["problem"] == "demand"
    assert body["contributions"]


def test_explain_demand_unknown_zone_returns_400(client):
    c, _ = client
    resp = c.post(
        "/explain/demand",
        json={"zone": 99999, "target_hour": "2024-01-31T20:00:00"},
    )
    assert resp.status_code == 400


def test_forecast_demand_endpoint(client):
    c, demand = client
    zone = int(demand["PULocationID"].iloc[0])
    resp = c.post(
        "/forecast/demand",
        json={"horizon_hours": 3, "zones": [zone]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon_hours"] == 3
    assert body["zones"] == 1
    # The in-process test service has no registry versions, so it reports the
    # configured reference.
    assert body["model_reference"] == "champion"
    assert body["request_id"]
    assert len(body["points"]) == 3
    for point in body["points"]:
        assert point["zone"] == zone
        assert point["predicted_pickups"] >= 0
        assert point["pickup_hour"]
    # Forecast hours are consecutive and strictly increasing.
    hours = [pd.Timestamp(p["pickup_hour"]) for p in body["points"]]
    assert hours == sorted(hours)


def test_forecast_demand_unknown_zone_returns_400(client):
    c, _ = client
    resp = c.post(
        "/forecast/demand",
        json={"horizon_hours": 2, "zones": [99999]},
    )
    assert resp.status_code == 400


FARE_BODY = {
    "trip_distance": 3.4,
    "pu_location_id": 1,
    "do_location_id": 2,
    "passenger_count": 1,
    "pickup_datetime": "2024-02-27T18:30:00",
}


def test_version_endpoint_reports_build_info(client, tmp_path, monkeypatch):
    from src.serving import api

    info = {
        "git_commit": "abc1234",
        "built_at": "2026-11-17T10:00:00+00:00",
        "models": [{"name": "taxi-fare-predictor", "version": 2, "run_id": "r1"}],
        "inputs": {"mlflow.db": "deadbeef"},
    }
    path = tmp_path / "build_info.json"
    path.write_text(json.dumps(info))
    monkeypatch.setattr(api, "BUILD_INFO_PATH", path)
    c, _ = client
    body = c.get("/version").json()
    assert body["git_commit"] == "abc1234"
    assert body["inputs"] == {"mlflow.db": "deadbeef"}
    assert body["fare_without_duration"] == "fixed duration estimate"
    assert body["demand_model_horizon_hours"] == 2


def test_api_key_protects_prediction_endpoints(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    c, _ = client
    assert c.post("/predict/fare", json=FARE_BODY).status_code == 401
    assert c.post("/predict/fare", json=FARE_BODY, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/predict/fare", json=FARE_BODY, headers={"X-API-Key": "s3cret"}).status_code == 200
    # Health stays open for load balancer checks.
    assert c.get("/health").status_code == 200


def test_rate_limit_returns_429_after_budget(client, monkeypatch):
    from src.serving.security import limiter

    limiter.reset()
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    c, _ = client
    codes = [c.post("/predict/fare", json=FARE_BODY).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert c.get("/health").status_code == 200
    limiter.reset()


def test_unknown_zone_fare_is_flagged_low_confidence(client):
    c, _ = client
    body = c.post("/predict/fare", json={**FARE_BODY, "do_location_id": 264}).json()
    assert body["low_confidence"] is True
    assert body["notes"]
    normal = c.post("/predict/fare", json=FARE_BODY).json()
    assert normal["low_confidence"] is False


def test_fare_response_names_the_model_that_answered(client, clean_trips_df):
    from sklearn.tree import DecisionTreeRegressor

    from src.features.fare_features import build_fare_features
    from src.models import common
    from src.serving import service as service_module

    c, _ = client
    assert c.post("/predict/fare", json=FARE_BODY).json()["duration_estimated"] is True

    Xb, yb = common.split_to_xy(build_fare_features(clean_trips_df), "fare_booking")
    svc = service_module._service
    svc.fare_booking_model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(Xb, yb)
    svc.model_versions = {
        "fare": {"name": "taxi-fare-predictor", "version": 2},
        "fare_booking": {"name": "taxi-fare-booking", "version": 1},
    }
    booking = c.post("/predict/fare", json=FARE_BODY).json()
    assert booking["model"] == "taxi-fare-booking/1"
    assert booking["duration_estimated"] is False
    full = c.post("/predict/fare", json={**FARE_BODY, "trip_duration_min": 14.0}).json()
    assert full["model"] == "taxi-fare-predictor/2"
    assert c.post("/explain/fare", json=FARE_BODY).status_code == 200


def test_forecast_points_report_their_method(client):
    c, demand = client
    zone = int(demand["PULocationID"].iloc[0])
    body = c.post("/forecast/demand", json={"horizon_hours": 4, "zones": [zone]}).json()
    assert body["model_horizon_hours"] == 2
    assert [p["method"] for p in body["points"]] == ["model", "model", "same_hour_last_week", "same_hour_last_week"]

    last = pd.Timestamp(demand["pickup_hour"].max())
    far = c.post("/predict/demand", json={"zone": zone, "target_hour": (last + pd.Timedelta(hours=5)).isoformat()})
    assert far.json()["method"] == "same_hour_last_week"
    explain_far = c.post(
        "/explain/demand", json={"zone": zone, "target_hour": (last + pd.Timedelta(hours=5)).isoformat()}
    )
    assert explain_far.status_code == 400


def test_api_key_also_protects_metrics_and_version_but_not_health(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    c, _ = client
    assert c.get("/health").status_code == 200
    for path in ("/metrics", "/version"):
        assert c.get(path).status_code == 401
        assert c.get(path, headers={"X-API-Key": "s3cret"}).status_code == 200
    schemes = c.get("/openapi.json").json()["components"]["securitySchemes"]
    assert schemes["APIKeyHeader"] == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
