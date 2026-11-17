"""Batch demand forecasting.

Produces a multi-hour-ahead pickup forecast for every known zone and writes the
results to SQLite.

Forecasts are routed by how far ahead they are:

- Up to ``serving.model_horizon_hours`` past the latest observed hour, the demand
  model forecasts *recursively*: each predicted hour is appended to the zone's
  history so the next hour's lag features see it.
- Beyond that, the forecast is the count at the same hour last week (seasonal
  naive). In the rolling backtest the model's recursive errors compound, and from
  three hours ahead the weekly pattern was more accurate.

Every row records which ``method`` produced it.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, Config, get_config
from src.serving import predict
from src.serving.service import ModelService

logger = logging.getLogger(__name__)

_TABLE = "demand_forecasts"
_WEEK = pd.Timedelta(days=7)
MODEL = "model"
SEASONAL_NAIVE = "same_hour_last_week"


def model_horizon(config: Config) -> int:
    """Hours past the latest observed hour that the demand model serves."""
    return int(config.serving.get("model_horizon_hours", 2))


def hours_ahead(target: pd.Timestamp, last_observed: pd.Timestamp) -> int:
    return int((pd.Timestamp(target) - pd.Timestamp(last_observed)) / pd.Timedelta(hours=1))


def seasonal_naive(observed: dict, target: pd.Timestamp, last_observed: pd.Timestamp) -> float:
    """Count at the same hour in the most recent observed week (0 if never seen)."""
    hour = pd.Timestamp(target) - _WEEK
    while hour > last_observed:
        hour -= _WEEK
    return float(observed.get(hour, 0.0))


def _resolve_db_path(config: Config) -> Path:
    path = Path(config.serving.get("batch_db_path", "predictions.db"))
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def forecast_demand(
    service: ModelService,
    *,
    horizon_hours: int | None = None,
    start_hour: pd.Timestamp | None = None,
    zones: list[int] | None = None,
    config: Config | None = None,
    model_hours: int | None = None,
) -> pd.DataFrame:
    """Forecast pickups for each zone over ``horizon_hours``.

    Returns a DataFrame of ``[PULocationID, pickup_hour, predicted_pickups, method]``.
    Forecasting starts at the hour following the latest history hour by default.
    ``model_hours`` overrides ``serving.model_horizon_hours`` (the evaluation uses
    it to score the model on its own at every horizon).
    """
    config = config or service.config
    horizon = horizon_hours or int(config.serving.get("batch_horizon_hours", 24))
    zones = zones or service.known_zones()
    model_hours = model_horizon(config) if model_hours is None else model_hours

    base_history = service.demand_history
    last_observed = service.latest_history_hour()
    if start_hour is None:
        start_hour = last_observed + pd.Timedelta(hours=1)
    start_hour = pd.Timestamp(start_hour).floor("h")

    records: list[dict] = []
    for zone in zones:
        # Per-zone running history we can append model predictions to.
        zone_hist = base_history[base_history["PULocationID"] == zone][
            ["PULocationID", "pickup_hour", "pickup_count"]
        ].copy()
        observed = dict(zip(pd.to_datetime(zone_hist["pickup_hour"]), zone_hist["pickup_count"], strict=True))
        for step in range(horizon):
            target = start_hour + pd.Timedelta(hours=step)
            if hours_ahead(target, last_observed) > model_hours:
                records.append(
                    {
                        "PULocationID": zone,
                        "pickup_hour": target,
                        "predicted_pickups": round(seasonal_naive(observed, target, last_observed), 4),
                        "method": SEASONAL_NAIVE,
                    }
                )
                continue
            pred = predict.predict_demand(
                service.demand_model, zone_hist, zone=zone, target_hour=target, config=config
            )
            records.append(
                {
                    "PULocationID": zone,
                    "pickup_hour": target,
                    "predicted_pickups": round(pred, 4),
                    "method": MODEL,
                }
            )
            # Append the prediction so subsequent lags can see it.
            zone_hist = pd.concat(
                [
                    zone_hist,
                    pd.DataFrame(
                        {
                            "PULocationID": [zone],
                            "pickup_hour": [target],
                            "pickup_count": [int(round(pred))],
                        }
                    ),
                ],
                ignore_index=True,
            )

    result = pd.DataFrame.from_records(
        records, columns=["PULocationID", "pickup_hour", "predicted_pickups", "method"]
    )
    logger.info(
        "Forecast %d hours x %d zones = %d rows from %s (model up to %d hours ahead)",
        horizon, len(zones), len(result), start_hour, model_hours,
    )
    return result


def write_forecasts(
    forecasts: pd.DataFrame,
    config: Config | None = None,
    *,
    model_version: str = "latest",
) -> Path:
    """Write the forecast rows to the SQLite predictions database."""
    config = config or get_config()
    db_path = _resolve_db_path(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC).isoformat()
    rows = forecasts.copy()
    rows["pickup_hour"] = rows["pickup_hour"].astype(str)
    rows["generated_at"] = generated_at
    rows["model_version"] = model_version

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                PULocationID INTEGER NOT NULL,
                pickup_hour TEXT NOT NULL,
                predicted_pickups REAL NOT NULL,
                generated_at TEXT NOT NULL,
                model_version TEXT NOT NULL,
                method TEXT
            )
            """
        )
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({_TABLE})")}
        if "method" not in columns:  # databases created before forecasts were routed
            conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN method TEXT")
        rows.to_sql(_TABLE, conn, if_exists="append", index=False)
    logger.info("Wrote %d forecast rows to %s", len(rows), db_path)
    return db_path


def run_batch(
    config: Config | None = None,
    *,
    horizon_hours: int | None = None,
    service: ModelService | None = None,
) -> dict:
    """Load models, forecast and persist; return a small summary dict."""
    config = config or get_config()
    service = service or ModelService.from_registry(config)
    forecasts = forecast_demand(service, horizon_hours=horizon_hours, config=config)
    db_path = write_forecasts(forecasts, config, model_version=service.model_version("demand"))
    return {
        "rows": len(forecasts),
        "zones": forecasts["PULocationID"].nunique(),
        "hours": forecasts["pickup_hour"].nunique(),
        "db_path": str(db_path),
    }
