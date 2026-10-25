"""Streamlit entry point for the taxi-forecasting showcase dashboard.

Run locally:
    streamlit run dashboard/app.py

Configure the backend with the ``API_BASE_URL`` environment variable
(defaults to ``http://localhost:8000``).
"""

from __future__ import annotations

import streamlit as st

from dashboard import sections
from dashboard.config import api_base_url

st.set_page_config(
    page_title="Taxi Forecasting - Showcase",
    page_icon="🚕",
    layout="wide",
)

_PAGES = {
    "Overview": sections.overview,
    "Demand forecast": sections.demand_forecast,
    "Fare prediction": sections.fare_prediction,
    "Explainability": sections.explainability,
    "Responsible AI": sections.responsible_ai,
    "Monitoring & KPIs": sections.monitoring,
}


def main() -> None:
    st.sidebar.title("🚕 Taxi Forecasting")
    choice = st.sidebar.radio("Section", list(_PAGES))
    st.sidebar.caption(f"API: `{api_base_url()}`")
    st.sidebar.caption("NYC taxi demand & fare forecasting · FastAPI serving.")
    _PAGES[choice]()


if __name__ == "__main__":
    main()
