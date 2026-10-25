"""Schema validation and cleaning for raw TLC trip records.

Raw TLC data contains corrupt rows (negative fares, zero-distance trips,
implausible passenger counts, timestamps outside the target month, etc.).
This module enforces a schema and applies configurable plausibility filters,
returning a cleaned DataFrame plus a report describing what was removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)

# Columns the downstream pipeline depends on.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount",
)


class SchemaError(ValueError):
    """Raised when required columns are missing from the input data."""


@dataclass
class CleaningReport:
    """Summary of how many rows were dropped at each cleaning stage."""

    input_rows: int = 0
    output_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def total_dropped(self) -> int:
        return self.input_rows - self.output_rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "total_dropped": self.total_dropped,
            "dropped": dict(self.dropped),
        }


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure all required columns are present, else raise ``SchemaError``."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def filter_to_period(
    df: pd.DataFrame, year: int, month: int, time_col: str = "tpep_pickup_datetime"
) -> tuple[pd.DataFrame, int]:
    """Keep only rows whose ``time_col`` falls within the target year/month.

    NYC TLC files contain a small number of stray records with timestamps far
    outside the nominal month (e.g. years like 2002 or 2098). Left in place,
    these explode the dense demand grid, so they are dropped here. Returns the
    filtered DataFrame and the number of rows removed.
    """
    ts = pd.to_datetime(df[time_col], errors="coerce")
    mask = (ts.dt.year == year) & (ts.dt.month == month)
    removed = int((~mask).sum())
    return df.loc[mask].reset_index(drop=True), removed


def clean_trips(
    df: pd.DataFrame,
    config: Config | None = None,
    *,
    year: int | None = None,
    month: int | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and clean raw trip records.

    Returns the cleaned DataFrame and a :class:`CleaningReport`. A derived
    ``trip_duration_min`` column is added for convenience. When ``year`` and
    ``month`` are provided, records outside that period are dropped first.
    """
    config = config or get_config()
    c = config.cleaning
    validate_schema(df)

    report = CleaningReport(input_rows=len(df))
    df = df.copy()

    # Parse datetimes defensively (already datetime in Parquet, but be safe).
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"], errors="coerce"
    )
    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"], errors="coerce"
    )

    def _drop(mask: pd.Series, label: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            report.dropped[label] = n
            df = df.loc[~mask]

    # Null timestamps / required keys.
    _drop(
        df["tpep_pickup_datetime"].isna() | df["tpep_dropoff_datetime"].isna(),
        "null_timestamps",
    )
    _drop(df["PULocationID"].isna() | df["DOLocationID"].isna(), "null_location")

    # Restrict to the target reporting period (drops stray out-of-range dates).
    if year is not None and month is not None:
        df, removed = filter_to_period(df, year, month)
        if removed:
            report.dropped["outside_target_period"] = removed
    # Trip duration in minutes (derived).
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    _drop(
        (df["trip_duration_min"] < c["min_trip_duration_min"])
        | (df["trip_duration_min"] > c["max_trip_duration_min"]),
        "trip_duration_out_of_range",
    )
    _drop(
        (df["trip_distance"] < c["min_trip_distance"])
        | (df["trip_distance"] > c["max_trip_distance"]),
        "trip_distance_out_of_range",
    )
    _drop(
        (df["fare_amount"] < c["min_fare"]) | (df["fare_amount"] > c["max_fare"]),
        "fare_out_of_range",
    )
    _drop(
        (df["passenger_count"].fillna(0) < c["min_passenger_count"])
        | (df["passenger_count"].fillna(0) > c["max_passenger_count"]),
        "passenger_count_out_of_range",
    )

    df = df.reset_index(drop=True)
    report.output_rows = len(df)
    logger.info(
        "Cleaned trips: %d -> %d rows (dropped %d)",
        report.input_rows,
        report.output_rows,
        report.total_dropped,
    )
    return df, report
