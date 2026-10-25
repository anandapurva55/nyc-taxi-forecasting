"""Batch demand forecasting.

Produces a multi-hour-ahead pickup forecast for every known zone and writes the
results to SQLite. Forecasting is *recursive*: each predicted hour is appended to
the zone's history so that the next hour's lag/rolling features incorporate prior
predictions (``lag_1`` of hour ``t+1`` is the prediction for hour ``t``).
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
) -> pd.DataFrame:
    """Recursively forecast pickups for each zone over ``horizon_hours``.

    Returns a DataFrame of ``[PULocationID, pickup_hour, predicted_pickups]``.
    Forecasting starts at the hour following the latest history hour by default.
    """
    config = config or service.config
    horizon = horizon_hours or int(config.serving.get("batch_horizon_hours", 24))
    zones = zones or service.known_zones()

    base_history = service.demand_history
    if start_hour is None:
        start_hour = service.latest_history_hour() + pd.Timedelta(hours=1)
    start_hour = pd.Timestamp(start_hour).floor("h")

    records: list[dict] = []
    for zone in zones:
        # Per-zone running history we can append predictions to.
        zone_hist = base_history[base_history["PULocationID"] == zone][
            ["PULocationID", "pickup_hour", "pickup_count"]
        ].copy()
        for step in range(horizon):
            target = start_hour + pd.Timedelta(hours=step)
            pred = predict.predict_demand(
                service.demand_model, zone_hist, zone=zone, target_hour=target, config=config
            )
            records.append(
                {
                    "PULocationID": zone,
                    "pickup_hour": target,
                    "predicted_pickups": round(pred, 4),
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

    result = pd.DataFrame.from_records(records)
    logger.info(
        "Forecast %d hours x %d zones = %d rows from %s",
        horizon, len(zones), len(result), start_hour,
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
                model_version TEXT NOT NULL
            )
            """
        )
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
    db_path = write_forecasts(forecasts, config)
    return {
        "rows": len(forecasts),
        "zones": forecasts["PULocationID"].nunique(),
        "hours": forecasts["pickup_hour"].nunique(),
        "db_path": str(db_path),
    }
