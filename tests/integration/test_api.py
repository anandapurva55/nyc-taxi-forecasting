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
    assert body["model_reference"] == "latest"
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

