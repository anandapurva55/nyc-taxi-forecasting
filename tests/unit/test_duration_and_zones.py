"""Unit tests for the learned duration estimate, zone groups and airport features."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import zones
from src.features.duration import DurationLookup, distance_band
from src.features.fare_features import build_fare_features
from src.serving.features import DEFAULT_MIN_PER_MILE, estimate_duration


def _trips(rows):
    return pd.DataFrame(rows, columns=["trip_distance", "trip_duration_min", "hour"])


def test_distance_band_edges():
    assert distance_band(0.5) == 0
    assert distance_band(1.0) == 0
    assert distance_band(1.5) == 1
    assert distance_band(50.0) == 5


def test_lookup_learns_rush_hour_slowdown():
    rows = [(2.0, 10.0, 8)] * 60 + [(2.0, 5.0, 3)] * 60  # 5 vs 2.5 min/mile
    lookup = DurationLookup.fit(_trips(rows))
    assert lookup.estimate(2.0, 8) == pytest.approx(10.0)
    assert lookup.estimate(2.0, 3) == pytest.approx(5.0)


def test_lookup_falls_back_to_overall_for_sparse_cells():
    rows = [(2.0, 6.0, 8)] * 60 + [(2.0, 20.0, 9)] * 5  # hour 9 below min_rows
    lookup = DurationLookup.fit(_trips(rows))
    assert "9:1" not in lookup.table
    assert lookup.estimate(2.0, 9) == pytest.approx(2.0 * lookup.overall)


def test_lookup_round_trips_through_dict():
    lookup = DurationLookup.fit(_trips([(3.0, 9.0, 12)] * 60))
    restored = DurationLookup.from_dict(lookup.to_dict())
    assert restored.estimate(3.0, 12) == pytest.approx(lookup.estimate(3.0, 12))


def test_estimate_duration_prefers_supplied_then_lookup_then_fallback():
    lookup = DurationLookup(table={"8:1": 4.0}, overall=3.5)
    assert estimate_duration(2.0, 7.0, 8, lookup) == 7.0
    assert estimate_duration(2.0, None, 8, lookup) == pytest.approx(8.0)
    assert estimate_duration(2.0, None) == pytest.approx(2.0 * DEFAULT_MIN_PER_MILE)


def test_zone_helpers():
    assert zones.is_airport_trip(132, 50)
    assert zones.is_jfk_manhattan(132, 230)
    assert zones.is_jfk_manhattan(230, 132)
    assert not zones.is_jfk_manhattan(132, 138)
    assert zones.has_unknown_zone(264, 1)
    assert not zones.has_unknown_zone(1, 2)


def test_fare_features_add_airport_flags():
    trips = pd.DataFrame(
        {
            "trip_distance": [17.0, 2.0],
            "passenger_count": [1, 2],
            "trip_duration_min": [40.0, 10.0],
            "PULocationID": [132, 48],
            "DOLocationID": [230, 50],
            "fare_amount": [70.0, 12.0],
            "tpep_pickup_datetime": pd.to_datetime(["2024-02-27 18:30", "2024-02-27 19:00"]),
        }
    )
    feats = build_fare_features(trips)
    assert feats["airport_trip"].tolist() == [1, 0]
    assert feats["jfk_manhattan"].tolist() == [1, 0]
