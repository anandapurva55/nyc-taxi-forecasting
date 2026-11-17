"""Cached HTTP helpers for talking to the prediction API from Streamlit."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from dashboard.config import api_base_url, demo_mode

_TIMEOUT = 60


@st.cache_resource
def _session() -> requests.Session:
    """Shared keep-alive session so repeated calls skip the TCP handshake."""
    session = requests.Session()
    if os.environ.get("API_KEY"):
        session.headers["X-API-Key"] = os.environ["API_KEY"]
    return session


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = _session().post(f"{api_base_url()}{path}", json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict[str, Any]:
    resp = _session().get(f"{api_base_url()}/health", timeout=_TIMEOUT)
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


# In the static demo, every call is answered from a recording of the real API.
if demo_mode():
    from dashboard import demo_backend as _recorded

    health = _recorded.health  # noqa: F811
    forecast_demand = _recorded.forecast_demand  # noqa: F811
    known_zones = _recorded.known_zones  # noqa: F811
    predict_fare = _recorded.predict_fare  # noqa: F811
    explain_fare = _recorded.explain_fare  # noqa: F811
    predict_demand = _recorded.predict_demand  # noqa: F811
    explain_demand = _recorded.explain_demand  # noqa: F811
