"""Booking-time trip duration estimate for fare requests without a duration.

The fare model is trained on true trip durations, but a caller asking for a fare
before the trip usually does not know the duration. Rather than a fixed minutes
per mile, this learns the median minutes per mile from the training data by hour
of day and trip-length band, which captures rush-hour traffic and the fact that
short trips are slower per mile than long ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Upper edges (miles) of the trip-length bands; the last band is open-ended.
DISTANCE_BANDS = (1.0, 2.0, 5.0, 10.0, 20.0)
FALLBACK_MIN_PER_MILE = 3.0
MIN_TRIP_DURATION_MIN = 1.0


def distance_band(trip_distance: float) -> int:
    """Index of the trip-length band for ``trip_distance`` miles."""
    return int(np.searchsorted(DISTANCE_BANDS, float(trip_distance), side="left"))


@dataclass
class DurationLookup:
    """Median minutes per mile keyed by ``(hour, distance_band)``."""

    table: dict[str, float] = field(default_factory=dict)
    overall: float = FALLBACK_MIN_PER_MILE

    @classmethod
    def fit(cls, trips: pd.DataFrame, min_rows: int = 50) -> DurationLookup:
        """Learn the lookup from trips with ``trip_distance``, ``trip_duration_min``, ``hour``."""
        df = trips[["trip_distance", "trip_duration_min", "hour"]].copy()
        df = df[df["trip_distance"] > 0.1]
        df["min_per_mile"] = df["trip_duration_min"] / df["trip_distance"]
        df["band"] = np.searchsorted(DISTANCE_BANDS, df["trip_distance"].to_numpy(), side="left")
        grouped = df.groupby(["hour", "band"])["min_per_mile"].agg(["median", "size"])
        grouped = grouped[grouped["size"] >= min_rows]
        table = {f"{int(h)}:{int(b)}": float(m) for (h, b), m in grouped["median"].items()}
        return cls(table=table, overall=float(df["min_per_mile"].median()))

    def estimate(self, trip_distance: float, hour: int) -> float:
        """Estimated duration in minutes for a trip of ``trip_distance`` miles at ``hour``."""
        key = f"{int(hour)}:{distance_band(trip_distance)}"
        min_per_mile = self.table.get(key, self.overall)
        return max(MIN_TRIP_DURATION_MIN, float(trip_distance) * min_per_mile)

    def to_dict(self) -> dict:
        return {"table": self.table, "overall": self.overall, "bands": list(DISTANCE_BANDS)}

    @classmethod
    def from_dict(cls, data: dict) -> DurationLookup:
        return cls(table={k: float(v) for k, v in data["table"].items()}, overall=float(data["overall"]))
