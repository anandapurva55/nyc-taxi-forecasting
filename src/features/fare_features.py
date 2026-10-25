"""Fare-prediction feature engineering.

Builds a per-trip supervised table for predicting ``fare_amount`` from trip
attributes available at (or near) pickup time: trip distance, passenger count,
pickup/drop-off zones and calendar signals derived from the pickup timestamp.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, get_config
from src.features.calendar import add_calendar_features

logger = logging.getLogger(__name__)

TARGET = "fare_amount"
TIME_COL = "tpep_pickup_datetime"

BASE_NUMERIC = ["trip_distance", "passenger_count", "trip_duration_min"]
BASE_CATEGORICAL = ["PULocationID", "DOLocationID"]


def build_fare_features(
    trips: pd.DataFrame, config: Config | None = None
) -> pd.DataFrame:
    """Build the supervised fare-prediction feature table.

    Parameters
    ----------
    trips:
        Cleaned trip records (output of :func:`clean_trips`).
    """
    config = config or get_config()

    # Deterministic column order: a set here would vary between processes (string
    # hash randomisation), causing train/serve feature-order skew for tree models.
    required = BASE_NUMERIC + BASE_CATEGORICAL + [TARGET, TIME_COL]
    missing = set(required) - set(trips.columns)
    if missing:
        raise ValueError(f"Missing columns for fare features: {sorted(missing)}")

    df = trips[required].copy()
    df = add_calendar_features(df, TIME_COL)

    df["passenger_count"] = df["passenger_count"].fillna(1).astype("int64")
    df = df.reset_index(drop=True)
    logger.info("Fare feature table: %d rows x %d cols", *df.shape)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (everything except target/time)."""
    exclude = {TARGET, TIME_COL}
    return [c for c in df.columns if c not in exclude]
