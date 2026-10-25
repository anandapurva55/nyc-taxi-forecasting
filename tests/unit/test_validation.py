"""Tests for data validation / cleaning."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import SchemaError, clean_trips, filter_to_period, validate_schema


def test_validate_schema_raises_on_missing_columns():
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_clean_removes_dirty_rows(raw_trips):
    clean, report = clean_trips(raw_trips)

    assert report.input_rows == len(raw_trips)
    assert report.output_rows == len(clean)
    assert report.total_dropped >= 3  # the three injected dirty rows

    # No invalid values remain.
    assert (clean["fare_amount"] >= 0).all()
    assert (clean["fare_amount"] <= 500).all()
    assert (clean["passenger_count"] >= 1).all()
    assert (clean["trip_distance"] <= 100).all()
    assert (clean["trip_duration_min"] > 0).all()


def test_clean_adds_trip_duration(clean_trips_df):
    assert "trip_duration_min" in clean_trips_df.columns
    assert clean_trips_df["trip_duration_min"].notna().all()


def test_report_as_dict_keys(raw_trips):
    _, report = clean_trips(raw_trips)
    d = report.as_dict()
    assert {"input_rows", "output_rows", "total_dropped", "dropped"} <= d.keys()


def test_filter_to_period_drops_out_of_range():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-01-15", "2024-01-20", "2002-05-01", "2098-01-01"]
            )
        }
    )
    filtered, removed = filter_to_period(df, 2024, 1)
    assert removed == 2
    assert len(filtered) == 2


def test_clean_with_period_filters_stray_dates(raw_trips):
    dirty = raw_trips.copy()
    stray = dirty.iloc[[0]].copy()
    stray["tpep_pickup_datetime"] = pd.Timestamp("2002-05-01")
    stray["tpep_dropoff_datetime"] = pd.Timestamp("2002-05-01 00:10:00")
    dirty = pd.concat([dirty, stray], ignore_index=True)

    clean, report = clean_trips(dirty, year=2024, month=1)
    assert report.dropped.get("outside_target_period", 0) >= 1
    assert (clean["tpep_pickup_datetime"].dt.year == 2024).all()
