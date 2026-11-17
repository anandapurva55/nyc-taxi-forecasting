"""Live accuracy of served demand predictions against realised pickup counts.

Input drift says the data changed; it does not say the model got worse. This
module closes that gap: captured demand predictions (zone + target hour) are
joined to the actual hourly pickup counts once they are published, and live
error is compared with the model's offline test baseline.

Actual counts come from the NYC TLC monthly trip files, which are published
about two months after the month ends. :func:`build_actuals` downloads the
months the captured predictions target and builds hourly counts with the same
code as the training pipeline; predictions for months not yet published are
reported as pending.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

from src.models.common import mae, rmse
from src.monitoring.performance import DEFAULT_TOLERANCE, any_degraded, performance_degradation
from src.responsible.fairness import wape

logger = logging.getLogger(__name__)


def target_months(captured: pd.DataFrame) -> list[tuple[int, int]]:
    """Distinct (year, month) of the hours the captured predictions are for."""
    if captured.empty or "target_hour" not in captured.columns:
        return []
    hours = pd.to_datetime(captured["target_hour"], errors="coerce").dropna()
    return sorted({(int(h.year), int(h.month)) for h in hours})


def build_actuals(
    months: list[tuple[int, int]], download: Callable | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Hourly pickup counts for each month, built from the published TLC files.

    ``download(year, month)`` returns the local path of that month's trip file
    (default: :func:`src.data.ingestion.download_trips`). Returns the combined
    hourly table and the status of each month: ``published`` or ``not published``.
    """
    from src.data.aggregation import aggregate_hourly_demand
    from src.data.validation import select_pickups

    if download is None:
        from src.data.ingestion import download_trips

        def download(year, month):
            return download_trips(year=year, month=month)

    frames, status = [], {}
    for year, month in months:
        key = f"{year:04d}-{month:02d}"
        try:
            path = download(year, month)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (403, 404):
                status[key] = "not published"
                logger.info("TLC data for %s is not published yet", key)
                continue
            raise
        trips = pd.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
        pickups, _ = select_pickups(trips, year=year, month=month)
        frames.append(aggregate_hourly_demand(pickups))
        status[key] = "published"
    columns = ["PULocationID", "pickup_hour", "pickup_count"]
    actuals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return actuals, status


def join_demand_outcomes(captured: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Match each captured prediction to the realised count for its zone and hour.

    ``captured`` needs ``zone``, ``target_hour`` and ``prediction``; ``actuals``
    is the hourly demand table (``PULocationID``, ``pickup_hour``, ``pickup_count``).
    Predictions for hours with no published actuals are dropped. When the same
    zone and hour was predicted more than once, the latest prediction is kept.
    Hours served by the same-hour-last-week rule are excluded, so live error is
    compared like for like with the model's test baseline.
    """
    needed = {"zone", "target_hour", "prediction"}
    if captured.empty or not needed.issubset(captured.columns):
        return pd.DataFrame(columns=["zone", "target_hour", "prediction", "actual"])
    if "method" in captured.columns:
        captured = captured[captured["method"].isna() | (captured["method"] == "model")]
    preds = captured[list(needed | ({"timestamp"} & set(captured.columns)))].copy()
    preds["zone"] = pd.to_numeric(preds["zone"], errors="coerce")
    preds["target_hour"] = pd.to_datetime(preds["target_hour"], errors="coerce").dt.floor("h")
    preds = preds.dropna(subset=["zone", "target_hour", "prediction"])
    if "timestamp" in preds.columns:
        preds = preds.sort_values("timestamp")
    preds = preds.drop_duplicates(["zone", "target_hour"], keep="last")

    truth = actuals[["PULocationID", "pickup_hour", "pickup_count"]].rename(
        columns={"PULocationID": "zone", "pickup_hour": "target_hour", "pickup_count": "actual"}
    )
    truth["target_hour"] = pd.to_datetime(truth["target_hour"])
    preds["zone"] = preds["zone"].astype(truth["zone"].dtype)
    return preds.merge(truth, on=["zone", "target_hour"], how="inner")[
        ["zone", "target_hour", "prediction", "actual"]
    ]


def evaluate_live_demand(
    captured: pd.DataFrame,
    actuals: pd.DataFrame,
    baseline: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
    actuals_status: dict[str, str] | None = None,
) -> dict:
    """Live error, degradation flags and coverage for captured demand predictions."""
    joined = join_demand_outcomes(captured, actuals)
    summary = {
        "problem": "demand",
        "captured": int(len(captured)),
        "matched": int(len(joined)),
        "tolerance": tolerance,
    }
    if actuals_status is not None:
        pending = {m for m, s in actuals_status.items() if s != "published"}
        summary["actuals_months"] = actuals_status
        summary["pending"] = 0
        if pending and "target_hour" in captured.columns:
            months = pd.to_datetime(captured["target_hour"], errors="coerce").dt.strftime("%Y-%m")
            summary["pending"] = int(months.isin(pending).sum())
    if joined.empty:
        return {**summary, "status": "no_matched_actuals", "degraded": False}

    y_true = joined["actual"].to_numpy(dtype=float)
    y_pred = joined["prediction"].to_numpy(dtype=float)
    current = {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "wape": wape(y_true, y_pred)}
    comparable = {k: v for k, v in baseline.items() if k in current}
    degradation = performance_degradation(current, comparable, tolerance)
    return {
        **summary,
        "status": "ok",
        "window": [str(joined["target_hour"].min()), str(joined["target_hour"].max())],
        "current": current,
        "baseline": comparable,
        "degradation": degradation,
        "degraded": any_degraded(degradation),
    }


def cloudwatch_metrics(summary: dict) -> list[dict]:
    """Shape a live-performance summary as CloudWatch ``put_metric_data`` entries."""
    metrics = [
        {"MetricName": "MatchedPredictions", "Value": float(summary.get("matched", 0)), "Unit": "Count"},
        {"MetricName": "PendingPredictions", "Value": float(summary.get("pending", 0)), "Unit": "Count"},
        {"MetricName": "PerformanceDegraded", "Value": float(bool(summary.get("degraded"))), "Unit": "Count"},
    ]
    for name, value in (summary.get("current") or {}).items():
        if value is not None and np.isfinite(value):
            metrics.append({"MetricName": f"Live{name.upper()}", "Value": float(value), "Unit": "None"})
    return metrics
