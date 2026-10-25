"""Aggregate cleaned trip records into an hourly per-zone demand series.

The demand-forecasting target is the number of pickups per (zone, hour). This
module groups cleaned trips by pickup zone and truncated pickup hour, producing
a dense series (zero-filled for hours with no pickups) suitable for time-series
feature engineering.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def aggregate_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Build a dense hourly pickup-count series per zone.

    Parameters
    ----------
    df:
        Cleaned trip records with ``tpep_pickup_datetime`` and ``PULocationID``.

    Returns
    -------
    DataFrame with columns ``[PULocationID, pickup_hour, pickup_count]``, dense
    across the full hourly range for every zone (missing hours filled with 0).
    """
    if df.empty:
        return pd.DataFrame(
            columns=["PULocationID", "pickup_hour", "pickup_count"]
        )

    work = df[["tpep_pickup_datetime", "PULocationID"]].copy()
    work["pickup_hour"] = work["tpep_pickup_datetime"].dt.floor("h")

    counts = (
        work.groupby(["PULocationID", "pickup_hour"])
        .size()
        .rename("pickup_count")
        .reset_index()
    )

    return _densify(counts)


def _densify(counts: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing (zone, hour) combinations with a zero pickup count."""
    full_range = pd.date_range(
        counts["pickup_hour"].min(),
        counts["pickup_hour"].max(),
        freq="h",
    )
    zones = counts["PULocationID"].unique()

    index = pd.MultiIndex.from_product(
        [zones, full_range], names=["PULocationID", "pickup_hour"]
    )
    dense = (
        counts.set_index(["PULocationID", "pickup_hour"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    dense["pickup_count"] = dense["pickup_count"].astype("int64")
    logger.info(
        "Aggregated demand: %d zones x %d hours = %d rows",
        len(zones),
        len(full_range),
        len(dense),
    )
    return dense.sort_values(["PULocationID", "pickup_hour"]).reset_index(
        drop=True
    )
