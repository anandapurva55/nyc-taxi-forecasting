"""The two fare models, running in the browser for the static demo.

Both fare models are LightGBM, which runs in browser Python (Pyodide). They are
exported from the MLflow registry as LightGBM's own model text by
``scripts/build_static_demo.py``. The feature row is built by the API's own
functions (``src.serving.predict``), and the responses have the same shape as
the API's, so the web app cannot tell the difference. SHAP values come from
LightGBM's built-in TreeSHAP (``pred_contrib``), because the ``shap`` package
is not available in the browser.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from dashboard.config import demo_models_dir

FARE_FIELDS = (
    "trip_distance", "passenger_count", "pu_location_id", "do_location_id",
    "pickup_datetime", "trip_duration_min",
)


class _Booster:
    """A LightGBM booster with the ``predict(X)`` interface the API's helpers expect."""

    def __init__(self, path) -> None:
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(path))
        self.features = self.booster.feature_name()

    def predict(self, X: pd.DataFrame, contributions: bool = False):
        return self.booster.predict(X[self.features], pred_contrib=contributions)


@lru_cache(maxsize=2)
def model(problem: str) -> _Booster:
    return _Booster(demo_models_dir() / f"{problem}.txt")


def problem_for(payload: dict[str, Any]) -> str:
    """Same rule as the API: without a trip duration, the booking-time model answers."""
    return "fare_booking" if payload.get("trip_duration_min") is None else "fare"


def _kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    kwargs = {k: payload.get(k) for k in FARE_FIELDS}
    kwargs["passenger_count"] = int(kwargs.get("passenger_count") or 1)
    kwargs["pickup_datetime"] = pd.Timestamp(kwargs["pickup_datetime"])
    return kwargs


def predict_fare(payload: dict[str, Any], model_names: dict[str, str]) -> dict[str, Any]:
    from src.features import zones
    from src.serving import predict

    problem = problem_for(payload)
    fare = predict.predict_fare(model(problem), booking=problem == "fare_booking", **_kwargs(payload))
    unknown = zones.has_unknown_zone(int(payload["pu_location_id"]), int(payload["do_location_id"]))
    return {
        "predicted_fare": round(fare, 2),
        "model": model_names.get(problem, problem),
        "duration_estimated": False,
        "low_confidence": unknown,
        "notes": [zones.UNKNOWN_ZONE_NOTE] if unknown else [],
        "request_id": "in-browser",
    }


def explain_fare(payload: dict[str, Any]) -> dict[str, Any]:
    from src.serving import predict

    problem = problem_for(payload)
    booster = model(problem)
    X = predict._fare_row(booking=problem == "fare_booking", config=_config(), **_kwargs(payload))
    values = booster.predict(X, contributions=True)[0]
    *contributions, base = (float(v) for v in values)
    ranked = sorted(zip(booster.features, contributions, strict=True), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "problem": "fare",
        "prediction": round(base + sum(contributions), 4),
        "base_value": round(base, 4),
        "contributions": [{"feature": f, "value": v} for f, v in ranked],
        "request_id": "in-browser",
    }


def _config():
    from src.config import get_config

    return get_config()
