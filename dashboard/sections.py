"""Rendering functions for each dashboard section."""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dashboard import api_client, data

# Assumed average fare per pickup, used only for the illustrative revenue KPI.
_AVG_FARE_USD = 18.0
# Horizon (hours) used for the interactive demand chart. A full city-wide,
# multi-hour recursive forecast over every zone is expensive, so the monitoring
# KPIs below use a fast single-hour, all-zone forecast instead.
_KPI_HORIZON = 1


def _handle_api_error(exc: Exception) -> None:
    st.error(
        "Could not reach the prediction API. Check that it is running and that "
        f"`API_BASE_URL` points at it.\n\n```\n{exc}\n```"
    )


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def overview() -> None:
    st.title("🚕 NYC Taxi Forecasting - MLOps Showcase")
    st.markdown(
        """
This platform serves **two models** trained on NYC TLC Yellow Taxi trips from
February 2024:

- **Demand forecaster** (XGBoost): hourly pickup counts per taxi zone, with
  recursive multi-hour forecasting.
- **Fare predictor** (LightGBM): trip fare from distance, zones, time and duration.

The system runs as eight connected stages: data pipeline → model training →
FastAPI serving → CI/CD and infrastructure → monitoring and drift detection →
responsible AI → this dashboard → one-command golden path. Every number on
these pages comes from the live API shown in the sidebar or from the generated
analysis reports.
        """
    )

    try:
        h = api_client.health()
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Service", h.get("status", "?").upper())
    c2.metric("Zones with history", h.get("zones", 0))
    c3.metric("History hours", h.get("history_hours", 0))
    c4.metric(
        "Models loaded",
        int(h.get("demand_model_loaded", False)) + int(h.get("fare_model_loaded", False)),
    )

    st.subheader("Headline model quality")
    cols = st.columns(2)
    for col, problem in zip(cols, ("demand", "fare"), strict=False):
        report = data.fairness_report(problem)
        with col:
            st.markdown(f"**{problem.title()} model**")
            if report:
                m = report["metrics"]
                st.metric("RMSE", f"{m['rmse']:.2f}")
                st.caption(
                    f"MAE {m['mae']:.2f} · MAPE {m['mape']:.1f}% · R² {m['r2']:.3f}"
                )
            else:
                st.info("Fairness report not available.")


# --------------------------------------------------------------------------- #
# Demand forecast
# --------------------------------------------------------------------------- #
def demand_forecast() -> None:
    st.title("📈 Demand Forecast")
    st.caption("Recursive multi-hour pickup forecasts via `POST /forecast/demand`.")

    try:
        zones = api_client.known_zones()
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return

    if not zones:
        st.warning("No zones with demand history are available.")
        return

    labels = data.zone_labels(zones)
    horizon = st.slider("Forecast horizon (hours)", 1, 72, 24)
    default = zones[: min(5, len(zones))]
    selected = st.multiselect(
        "Zones",
        options=zones,
        default=default,
        format_func=lambda z: labels.get(z, str(z)),
    )
    if not selected:
        st.info("Select at least one zone.")
        return

    try:
        response = api_client.forecast_demand(horizon, tuple(sorted(selected)))
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return

    points = response["points"]
    df = pd.DataFrame(points)
    if df.empty:
        st.warning("No forecast returned.")
        return
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    df = data.enrich_with_zones(df)
    df["label"] = df["zone"].map(labels)

    fig = px.line(
        df,
        x="pickup_hour",
        y="predicted_pickups",
        color="label",
        markers=True,
        labels={
            "pickup_hour": "Hour",
            "predicted_pickups": "Predicted pickups",
            "label": "Zone",
        },
        title="Predicted pickups over the forecast horizon",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Request `{response['request_id']}` · model reference "
        f"`{response.get('model_reference', 'not reported')}`"
    )

    st.subheader("Total predicted pickups by zone")
    totals = (
        df.groupby("label", as_index=False)["predicted_pickups"]
        .sum()
        .sort_values("predicted_pickups", ascending=False)
    )
    bar = px.bar(
        totals,
        x="predicted_pickups",
        y="label",
        orientation="h",
        labels={"predicted_pickups": "Total predicted pickups", "label": "Zone"},
    )
    st.plotly_chart(bar, use_container_width=True)


# --------------------------------------------------------------------------- #
# Fare prediction + explanation
# --------------------------------------------------------------------------- #
def fare_prediction() -> None:
    st.title("💵 Fare Prediction")
    st.caption("`POST /predict/fare` with a live SHAP explanation via `/explain/fare`.")

    with st.form("fare_form"):
        c1, c2 = st.columns(2)
        trip_distance = c1.number_input("Trip distance (mi)", 0.1, 100.0, 3.4, 0.1)
        passenger_count = c2.number_input("Passengers", 1, 8, 1)
        pu = c1.number_input("Pickup zone ID", 1, 265, 132)
        do = c2.number_input("Drop-off zone ID", 1, 265, 230)
        pickup_date = c1.date_input("Pickup date", datetime(2024, 2, 27))
        pickup_time = c2.time_input("Pickup time", time(18, 30))
        submitted = st.form_submit_button("Predict fare")

    if not submitted:
        return

    payload = {
        "trip_distance": float(trip_distance),
        "pu_location_id": int(pu),
        "do_location_id": int(do),
        "passenger_count": int(passenger_count),
        "pickup_datetime": datetime.combine(pickup_date, pickup_time).isoformat(),
    }
    try:
        pred = api_client.predict_fare(payload)
        explanation = api_client.explain_fare(payload)
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return

    st.metric("Predicted fare", f"${pred['predicted_fare']:.2f}")
    if pred.get("duration_estimated"):
        st.caption("Trip duration was estimated from distance.")

    _render_contributions(explanation, "Fare - feature contributions (SHAP)")


# --------------------------------------------------------------------------- #
# Explainability (demand)
# --------------------------------------------------------------------------- #
def explainability() -> None:
    st.title("🔍 Explainability")
    st.caption("SHAP feature contributions for a single demand prediction.")

    try:
        zones = api_client.known_zones()
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return
    if not zones:
        st.warning("No zones with demand history are available.")
        return

    labels = data.zone_labels(zones)
    zone = st.selectbox("Zone", zones, format_func=lambda z: labels.get(z, str(z)))
    c1, c2 = st.columns(2)
    target_date = c1.date_input("Target date", datetime(2024, 2, 29))
    target_time = c2.time_input("Target hour", time(20, 0))
    target = datetime.combine(target_date, target_time).isoformat()

    if not st.button("Explain demand prediction"):
        return
    try:
        explanation = api_client.explain_demand(int(zone), target)
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return

    st.metric("Predicted pickups", f"{explanation['prediction']:.2f}")
    _render_contributions(explanation, "Demand - feature contributions (SHAP)")


def _render_contributions(explanation: dict, title: str) -> None:
    contribs = pd.DataFrame(explanation["contributions"])
    if contribs.empty:
        st.info("No contributions returned.")
        return
    contribs = contribs.sort_values("value", key=lambda s: s.abs())
    fig = px.bar(
        contribs,
        x="value",
        y="feature",
        orientation="h",
        color="value",
        color_continuous_scale="RdBu",
        labels={"value": "SHAP contribution", "feature": "Feature"},
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Base value {explanation['base_value']:.3f} + contributions ≈ "
        f"prediction {explanation['prediction']:.3f} (SHAP additivity)."
    )


# --------------------------------------------------------------------------- #
# Responsible AI
# --------------------------------------------------------------------------- #
def responsible_ai() -> None:
    st.title("⚖️ Responsible AI")
    problem = st.radio("Model", ("demand", "fare"), horizontal=True)

    report = data.fairness_report(problem)
    if report:
        st.subheader("Fairness across subgroups")
        for dim in report["dimensions"]:
            flag = "🚩 flagged" if dim["flagged"] else "✓ within threshold"
            st.markdown(
                f"**{dim['dimension']}** - disparity ratio "
                f"{dim['disparity_ratio']:.2f} (threshold {dim['threshold']}) · {flag}"
            )
            sub = pd.DataFrame(dim["subgroups"]).sort_values("rmse", ascending=False)
            st.dataframe(
                sub[["group", "n", "rmse", "mae", "mape"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Fairness report not available.")

    shap_png = data.shap_summary_path(problem)
    if shap_png:
        st.subheader("Global SHAP summary")
        st.image(str(shap_png), use_column_width=True)

    card = data.model_card(problem)
    if card:
        with st.expander("📄 Model card", expanded=False):
            st.markdown(card)


# --------------------------------------------------------------------------- #
# Monitoring & business KPIs
# --------------------------------------------------------------------------- #
def monitoring() -> None:
    st.title("📊 Monitoring & Business KPIs")
    st.caption(
        "Illustrative KPIs derived from a live next-hour demand forecast across "
        "all zones. Drift reports are generated separately."
    )

    try:
        response = api_client.forecast_demand(_KPI_HORIZON, None)
    except (requests.RequestException, ValueError) as exc:
        _handle_api_error(exc)
        return

    points = response["points"]
    df = pd.DataFrame(points)
    if df.empty:
        st.warning("No forecast returned.")
        return
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    df = data.enrich_with_zones(df)
    total = df["predicted_pickups"].sum()
    next_hour = pd.Timestamp(df["pickup_hour"].min()).strftime("%Y-%m-%d %H:%M")
    active_zones = int((df["predicted_pickups"] > 0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted pickups (next hour)", f"{total:,.0f}")
    c2.metric("Est. revenue (next hour)", f"${total * _AVG_FARE_USD:,.0f}")
    c3.metric("Active zones", active_zones)
    st.caption(f"Forecast hour: {next_hour}")
    st.caption(
        f"Request `{response['request_id']}` · model reference "
        f"`{response.get('model_reference', 'not reported')}`"
    )

    by_borough = (
        df.groupby("Borough", as_index=False)["predicted_pickups"]
        .sum()
        .sort_values("predicted_pickups", ascending=False)
    )
    fig = px.bar(
        by_borough,
        x="Borough",
        y="predicted_pickups",
        labels={"predicted_pickups": "Predicted pickups"},
        title="Predicted pickups by borough (next hour)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top zones (next hour)")
    top = df.sort_values("predicted_pickups", ascending=False).head(15)
    st.dataframe(
        top[["zone", "Zone", "Borough", "predicted_pickups"]],
        use_container_width=True,
        hide_index=True,
    )

    _render_drift_panel()


def _render_drift_panel() -> None:
    """Render the latest feature-drift reports, if any are present."""
    st.markdown("---")
    st.subheader("🔎 Feature drift")

    reports = data.drift_reports()
    if not reports:
        st.info(
            "No drift reports found. Generate one (no AWS needed) with:\n\n"
            "```\npython -m scripts.monitoring.demo_drift_report --problem fare "
            "--mode drift\n```\n\n"
            "In production this JSON is produced by the scheduled SageMaker drift "
            "job (`scripts/monitoring/run_drift_job.py`) and drives CloudWatch "
            "alarms → SNS alerts → retraining."
        )
        return

    st.caption(
        "Live view of the drift job's output (PSI ≥ threshold ⇒ drifted). Same "
        "report the scheduled SageMaker job writes to S3 in production."
    )

    for report in reports:
        problem = report.get("problem", "unknown")
        drifted = bool(report.get("drifted"))
        badge = "🔴 DRIFT DETECTED" if drifted else "🟢 No drift"
        n_drifted = report.get("n_drifted", 0)
        n_features = report.get("n_features", 0)
        source = report.get("_source", "")
        updated = (
            datetime.fromtimestamp(report["_mtime"]).strftime("%Y-%m-%d %H:%M")
            if report.get("_mtime")
            else "n/a"
        )

        st.markdown(f"**{problem}** - {badge}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Features drifted", f"{n_drifted}/{n_features}")
        c2.metric("Share drifted", f"{report.get('share_drifted', 0.0):.0%}")
        c3.metric("Records evaluated", f"{report.get('n_current', 0):,}")

        features = report.get("features") or []
        if features:
            fdf = pd.DataFrame(features).sort_values("psi", ascending=False)
            threshold = report.get("psi_threshold", 0.2)
            fig = px.bar(
                fdf,
                x="psi",
                y="feature",
                orientation="h",
                color="drifted",
                color_discrete_map={True: "#d62728", False: "#1f77b4"},
                labels={"psi": "PSI", "feature": "Feature", "drifted": "Drifted"},
                title=f"Population Stability Index by feature - {problem}",
            )
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"threshold {threshold}",
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Per-feature detail (PSI / KS)", expanded=False):
                st.dataframe(
                    fdf[["feature", "psi", "ks_statistic", "ks_pvalue", "drifted"]],
                    use_container_width=True,
                    hide_index=True,
                )
        elif report.get("status") == "no_data":
            st.warning("No captured records were available for this problem.")

        st.caption(f"Source: `{source}` · updated {updated}")

    st.info(
        "This panel reads local drift-job reports. Cloud alerts and scheduled "
        "retraining require separate configuration; drift does not automatically "
        "start retraining or authorize a deployment."
    )
