"""Recorded API for the static demo on GitHub Pages.

GitHub Pages serves files only, so the demo has no API behind it. This module
answers the web app's API calls. Demand forecasts, health and demand
explanations come from ``demo/recording.json``, which
``scripts/build_static_demo.py`` writes by calling the real API, with the real
registered models, once; demand requests that were not recorded are refused.
Fare quotes run the two real fare models in the browser (``demo_models``), so
any trip can be priced.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from dashboard.config import demo_recording_path


class NotRecorded(ValueError):
    """The demo only answers requests that were recorded from the real API."""


@lru_cache(maxsize=1)
def recording() -> dict[str, Any]:
    return json.loads(demo_recording_path().read_text(encoding="utf-8"))


def health() -> dict[str, Any]:
    return recording()["health"]


def build() -> dict[str, Any]:
    """Commit, models and time of the recording, shown in the banner and sidebar."""
    return recording()["build"]


def known_zones() -> list[int]:
    return sorted(int(z) for z in recording()["forecast"]["zones"])


def forecast_demand(horizon_hours: int, zones: tuple[int, ...] | None = None) -> dict[str, Any]:
    rec = recording()["forecast"]
    hours = rec["hours"][: max(1, min(int(horizon_hours), len(rec["hours"])))]
    selected = list(zones) if zones else known_zones()
    points = [
        {
            "zone": int(zone),
            "pickup_hour": hour,
            "predicted_pickups": rec["zones"][str(zone)][i],
            "method": rec["methods"][i],
        }
        for zone in selected
        if str(zone) in rec["zones"]
        for i, hour in enumerate(hours)
    ]
    return {
        "horizon_hours": len(hours),
        "model_horizon_hours": rec["model_horizon_hours"],
        "zones": len({p["zone"] for p in points}),
        "points": points,
        "model_reference": rec["model_reference"],
        "request_id": "recorded",
    }


def fare_samples() -> list[dict[str, Any]]:
    """Fare quotes recorded from the API, which the browser's models must reproduce."""
    return recording()["fare_samples"]


def predict_fare(payload: dict[str, Any]) -> dict[str, Any]:
    from dashboard import demo_models

    return demo_models.predict_fare(payload, build()["models"])


def explain_fare(payload: dict[str, Any]) -> dict[str, Any]:
    from dashboard import demo_models

    return demo_models.explain_fare(payload)


def demand_samples() -> list[dict[str, Any]]:
    return recording()["demand_samples"]


def _demand_sample(zone: int, target_hour: str) -> dict[str, Any]:
    for sample in demand_samples():
        if sample["zone"] == int(zone) and sample["target_hour"] == target_hour:
            return sample
    raise NotRecorded("Only the sample zones and hours were recorded for the demo.")


def predict_demand(zone: int, target_hour: str) -> dict[str, Any]:
    return _demand_sample(zone, target_hour)["prediction"]


def explain_demand(zone: int, target_hour: str) -> dict[str, Any]:
    return _demand_sample(zone, target_hour)["explanation"]
