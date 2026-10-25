"""Tests for the thin Python API client.

The client is exercised against the in-process FastAPI ``TestClient`` (which is
API-compatible with the ``requests`` methods the client uses), so no running
server is required.
"""

from __future__ import annotations

import pandas as pd

from clients.python.taxi_client import TaxiClient


def _client(test_client) -> TaxiClient:
    # Empty base_url so paths like "/health" are sent as-is to the TestClient.
    return TaxiClient(base_url="", session=test_client)


def test_client_health(client):
    c, _ = client
    body = _client(c).health()
    assert body["status"] == "ok"


def test_client_predict_fare(client):
    c, _ = client
    body = _client(c).predict_fare(
        trip_distance=3.4,
        pu_location_id=1,
        do_location_id=2,
        pickup_datetime="2024-01-15T18:30:00",
    )
    assert body["predicted_fare"] >= 0


def test_client_forecast_demand(client):
    c, demand = client
    zone = int(demand["PULocationID"].iloc[0])
    body = _client(c).forecast_demand(horizon_hours=2, zones=[zone])
    assert body["horizon_hours"] == 2
    assert len(body["points"]) == 2


def test_client_predict_demand(client):
    c, demand = client
    zone = int(demand["PULocationID"].iloc[0])
    target = pd.Timestamp(demand["pickup_hour"].max())
    body = _client(c).predict_demand(zone=zone, target_hour=target)
    assert body["zone"] == zone
    assert body["predicted_pickups"] >= 0
