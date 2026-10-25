"""Thin Python client for the taxi-forecasting prediction API.

A minimal ``requests``-based wrapper around the FastAPI service so notebooks,
the Streamlit dashboard and downstream jobs can call predictions without
hand-rolling HTTP. Every method raises ``requests.HTTPError`` on non-2xx
responses and returns the parsed JSON body.

Example
-------
    from clients.python.taxi_client import TaxiClient

    client = TaxiClient("http://localhost:8000")
    client.health()
    client.predict_fare(trip_distance=3.4, pu_location_id=132,
                        do_location_id=230, pickup_datetime="2024-01-15T18:30:00")
    client.forecast_demand(horizon_hours=6, zones=[132, 230])
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

DEFAULT_TIMEOUT = 30


class TaxiClient:
    """Client for the taxi-forecasting FastAPI service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    # -- internal helpers ----------------------------------------------------
    def _get(self, path: str) -> dict[str, Any]:
        resp = self._session.get(f"{self.base_url}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}{path}", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _iso(value: str | datetime) -> str:
        return value.isoformat() if isinstance(value, datetime) else value

    # -- endpoints -----------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """Return service liveness + model/state readiness."""
        return self._get("/health")

    def predict_fare(
        self,
        *,
        trip_distance: float,
        pu_location_id: int,
        do_location_id: int,
        pickup_datetime: str | datetime,
        passenger_count: int = 1,
        trip_duration_min: float | None = None,
    ) -> dict[str, Any]:
        """Predict the fare for a single trip."""
        payload: dict[str, Any] = {
            "trip_distance": trip_distance,
            "pu_location_id": pu_location_id,
            "do_location_id": do_location_id,
            "passenger_count": passenger_count,
            "pickup_datetime": self._iso(pickup_datetime),
        }
        if trip_duration_min is not None:
            payload["trip_duration_min"] = trip_duration_min
        return self._post("/predict/fare", payload)

    def predict_demand(
        self, *, zone: int, target_hour: str | datetime
    ) -> dict[str, Any]:
        """Predict the pickup count for a single ``(zone, hour)``."""
        return self._post(
            "/predict/demand",
            {"zone": zone, "target_hour": self._iso(target_hour)},
        )

    def forecast_demand(
        self,
        *,
        horizon_hours: int = 24,
        zones: list[int] | None = None,
        start_hour: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Recursive multi-hour pickup forecast for one or more zones."""
        payload: dict[str, Any] = {"horizon_hours": horizon_hours}
        if zones is not None:
            payload["zones"] = zones
        if start_hour is not None:
            payload["start_hour"] = self._iso(start_hour)
        return self._post("/forecast/demand", payload)

    def explain_fare(self, **kwargs: Any) -> dict[str, Any]:
        """Return the SHAP explanation for a fare prediction.

        Accepts the same keyword arguments as :meth:`predict_fare`.
        """
        payload = {
            "trip_distance": kwargs["trip_distance"],
            "pu_location_id": kwargs["pu_location_id"],
            "do_location_id": kwargs["do_location_id"],
            "passenger_count": kwargs.get("passenger_count", 1),
            "pickup_datetime": self._iso(kwargs["pickup_datetime"]),
        }
        if kwargs.get("trip_duration_min") is not None:
            payload["trip_duration_min"] = kwargs["trip_duration_min"]
        return self._post("/explain/fare", payload)

    def explain_demand(
        self, *, zone: int, target_hour: str | datetime
    ) -> dict[str, Any]:
        """Return the SHAP explanation for a demand prediction."""
        return self._post(
            "/explain/demand",
            {"zone": zone, "target_hour": self._iso(target_hour)},
        )


def _demo(base_url: str = "http://localhost:8000") -> None:
    """Small end-to-end demo against a running service."""
    client = TaxiClient(base_url)
    print("health:", client.health())
    print(
        "fare:",
        client.predict_fare(
            trip_distance=3.4,
            pu_location_id=132,
            do_location_id=230,
            pickup_datetime="2024-01-15T18:30:00",
        ),
    )
    print("forecast:", client.forecast_demand(horizon_hours=3, zones=[132]))


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
