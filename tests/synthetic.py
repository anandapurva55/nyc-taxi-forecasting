"""Synthetic NYC-TLC-shaped trip data for tests and offline development.

Generates a DataFrame with the same schema as the real Yellow Taxi Parquet
files so the pipeline can be exercised without network access. Includes a few
deliberately corrupt rows so cleaning logic has something to remove.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_trips(
    n_rows: int = 5000,
    n_zones: int = 8,
    start: str = "2024-01-01",
    days: int = 14,
    seed: int = 42,
    include_dirty: bool = True,
) -> pd.DataFrame:
    """Return a synthetic trips DataFrame matching the TLC schema."""
    rng = np.random.default_rng(seed)

    start_ts = pd.Timestamp(start)
    span_seconds = days * 24 * 3600
    offsets = rng.integers(0, span_seconds, size=n_rows)
    pickup = start_ts + pd.to_timedelta(np.sort(offsets), unit="s")

    distance = np.round(rng.gamma(2.0, 1.5, size=n_rows), 2)
    duration_min = np.clip(distance * rng.uniform(2, 5, size=n_rows), 1, 180)
    dropoff = pickup + pd.to_timedelta(duration_min * 60, unit="s")

    fare = np.round(2.5 + distance * 2.8 + duration_min * 0.35, 2)

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "passenger_count": rng.integers(1, 5, size=n_rows).astype("float64"),
            "trip_distance": distance,
            "PULocationID": rng.integers(1, n_zones + 1, size=n_rows),
            "DOLocationID": rng.integers(1, n_zones + 1, size=n_rows),
            "fare_amount": fare,
            "total_amount": np.round(fare + rng.uniform(0, 5, size=n_rows), 2),
        }
    )

    if include_dirty:
        dirty = pd.DataFrame(
            {
                "tpep_pickup_datetime": [start_ts, start_ts, start_ts],
                "tpep_dropoff_datetime": [
                    start_ts + pd.Timedelta(minutes=10),
                    start_ts - pd.Timedelta(minutes=5),  # negative duration
                    start_ts + pd.Timedelta(minutes=10),
                ],
                "passenger_count": [0.0, 2.0, 2.0],  # zero passengers
                "trip_distance": [1.0, 2.0, 999.0],  # huge distance
                "PULocationID": [1, 2, 3],
                "DOLocationID": [2, 3, 4],
                "fare_amount": [-5.0, 10.0, 9999.0],  # negative & huge fare
                "total_amount": [-3.0, 12.0, 10000.0],
            }
        )
        df = pd.concat([df, dirty], ignore_index=True)

    return df
