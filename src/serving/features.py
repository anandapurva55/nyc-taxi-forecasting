"""Build model-input feature rows for online inference.

These helpers reuse the feature engineering (``build_demand_features`` /
``build_fare_features``) so the columns produced at serving time exactly match
those seen during training - avoiding train/serve skew.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, get_config
from src.features import demand_features, fare_features

logger = logging.getLogger(__name__)

# Fallback trip duration estimate when the caller does not supply one.
# (Booking-time callers usually don't know the duration; the fare model was
# trained with it, so we approximate from distance.)
DEFAULT_MIN_PER_MILE = 3.0
MIN_TRIP_DURATION_MIN = 1.0


def _max_history_hours(config: Config) -> int:
    lags = list(config.features.get("demand_lags", [1, 24, 168]))
    windows = list(config.features.get("demand_rolling_windows", [3, 24]))
    return max(max(lags), max(windows))


def build_demand_row(
    history: pd.DataFrame,
    zone: int,
    target_hour: pd.Timestamp,
    config: Config | None = None,
) -> pd.DataFrame:
    """Build the 20-column demand feature row for ``(zone, target_hour)``.

    ``history`` is the dense hourly pickup series
    (``[PULocationID, pickup_hour, pickup_count]``). Lag/rolling features use only
    past (shifted) values, so the unknown target-hour count is irrelevant.
    """
    config = config or get_config()
    target_hour = pd.Timestamp(target_hour).floor("h")

    zone_hist = history[history["PULocationID"] == zone][
        ["pickup_hour", "pickup_count"]
    ].copy()
    zone_hist["pickup_hour"] = pd.to_datetime(zone_hist["pickup_hour"])

    span = _max_history_hours(config)
    start = target_hour - pd.Timedelta(hours=span)
    full_range = pd.date_range(start, target_hour, freq="h")

    series = (
        zone_hist.set_index("pickup_hour")["pickup_count"]
        .reindex(full_range, fill_value=0)
    )
    dense = pd.DataFrame(
        {
            "PULocationID": zone,
            "pickup_hour": full_range,
            "pickup_count": series.to_numpy(),
        }
    )
    dense["pickup_count"] = dense["pickup_count"].astype("int64")

    feats = demand_features.build_demand_features(dense, config, dropna=False)
    row = feats[feats["pickup_hour"] == target_hour]
    if row.empty:  # pragma: no cover - defensive
        raise ValueError(f"Could not build features for hour {target_hour}")
    cols = demand_features.feature_columns(feats)
    return row[cols].reset_index(drop=True)


def estimate_duration(trip_distance: float, trip_duration_min: float | None) -> float:
    """Return a usable trip duration, estimating from distance when missing."""
    if trip_duration_min is not None:
        return float(trip_duration_min)
    return max(MIN_TRIP_DURATION_MIN, float(trip_distance) * DEFAULT_MIN_PER_MILE)


def build_fare_row(
    *,
    trip_distance: float,
    passenger_count: int,
    pu_location_id: int,
    do_location_id: int,
    pickup_datetime: pd.Timestamp,
    trip_duration_min: float | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Build the 14-column fare feature row for a single trip request."""
    config = config or get_config()
    duration = estimate_duration(trip_distance, trip_duration_min)

    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp(pickup_datetime)],
            "trip_distance": [float(trip_distance)],
            "trip_duration_min": [duration],
            "passenger_count": [int(passenger_count)],
            "PULocationID": [int(pu_location_id)],
            "DOLocationID": [int(do_location_id)],
            "fare_amount": [0.0],  # dummy target; dropped from model inputs.
        }
    )
    feats = fare_features.build_fare_features(raw, config)
    cols = fare_features.feature_columns(feats)
    return feats[cols].reset_index(drop=True)
