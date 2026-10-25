"""Cached HTTP helpers for talking to the prediction API from Streamlit."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from dashboard.config import api_base_url

_TIMEOUT = 60


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{api_base_url()}{path}", json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict[str, Any]:
    resp = requests.get(f"{api_base_url()}/health", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300, show_spinner="Forecasting demand…")
def forecast_demand(
    horizon_hours: int, zones: tuple[int, ...] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"horizon_hours": horizon_hours}
    if zones:
        payload["zones"] = list(zones)
    return _post("/forecast/demand", payload)


@st.cache_data(ttl=300, show_spinner=False)
def known_zones() -> list[int]:
    """All zones the demand model has history for (via a 1-hour forecast)."""
    points = forecast_demand(1, None)["points"]
    return sorted({int(p["zone"]) for p in points})


def predict_fare(payload: dict[str, Any]) -> dict[str, Any]:
    return _post("/predict/fare", payload)


def explain_fare(payload: dict[str, Any]) -> dict[str, Any]:
    return _post("/explain/fare", payload)


def predict_demand(zone: int, target_hour: str) -> dict[str, Any]:
    return _post("/predict/demand", {"zone": zone, "target_hour": target_hour})


def explain_demand(zone: int, target_hour: str) -> dict[str, Any]:
    return _post("/explain/demand", {"zone": zone, "target_hour": target_hour})
