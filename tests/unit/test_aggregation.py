"""Tests for hourly demand aggregation."""

from __future__ import annotations

from src.data.aggregation import aggregate_hourly_demand


def test_aggregate_columns_and_density(clean_trips_df):
    demand = aggregate_hourly_demand(clean_trips_df)

    assert list(demand.columns) == ["PULocationID", "pickup_hour", "pickup_count"]
    assert (demand["pickup_count"] >= 0).all()

    # Dense: every zone covers the same set of hours.
    counts_per_zone = demand.groupby("PULocationID")["pickup_hour"].nunique()
    assert counts_per_zone.nunique() == 1


def test_aggregate_total_matches_input(clean_trips_df):
    demand = aggregate_hourly_demand(clean_trips_df)
    assert demand["pickup_count"].sum() == len(clean_trips_df)


def test_aggregate_empty_input_returns_empty():
    import pandas as pd

    empty = pd.DataFrame(
        columns=["tpep_pickup_datetime", "PULocationID"]
    )
    result = aggregate_hourly_demand(empty)
    assert result.empty
