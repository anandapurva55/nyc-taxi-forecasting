"""Streamlit entry point for the taxi-forecasting showcase dashboard.

Run locally:
    streamlit run dashboard/app.py

Configure the backend with the ``API_BASE_URL`` environment variable
(defaults to ``http://localhost:8000``).
"""

from __future__ import annotations

import streamlit as st

from dashboard import sections
from dashboard.config import api_base_url, demo_mode

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


README_URL = "https://github.com/anandapurva55/nyc-taxi-forecasting#readme"


def _demo_banner() -> None:
    """Shown on every page of the static demo, so nobody mistakes it for the live service."""
    from dashboard import demo_backend

    build = demo_backend.build()
    st.info(
        "**Interactive demo.** There is no server here, to avoid cost. Fare quotes run the "
        "two real fare models in your browser, so you can price any trip. Demand forecasts "
        "and explanations are responses recorded from the real API on the February 2024 "
        f"models (commit `{build['git_commit']}`). The live system on AWS, with ECS Fargate, "
        f"CloudFront and SageMaker monitoring, is described in the [README]({README_URL}).",
        icon="ℹ️",
    )


def main() -> None:
    st.sidebar.title("🚕 Taxi Forecasting")
    choice = st.sidebar.radio("Section", list(_PAGES))
    if demo_mode():
        from dashboard import demo_backend

        st.sidebar.caption(
            f"Recorded from the live API at commit `{demo_backend.build()['git_commit']}`."
        )
        _demo_banner()
    else:
        st.sidebar.caption(f"API: `{api_base_url()}`")
    st.sidebar.caption("NYC taxi demand & fare forecasting · FastAPI serving.")
    _PAGES[choice]()


if __name__ == "__main__":
    main()
