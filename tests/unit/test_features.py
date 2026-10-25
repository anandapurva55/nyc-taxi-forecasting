"""Tests for demand and fare feature engineering."""

from __future__ import annotations

from src.data.aggregation import aggregate_hourly_demand
from src.features.demand_features import (
    build_demand_features,
)
from src.features.demand_features import (
    feature_columns as demand_feature_columns,
)
from src.features.fare_features import (
    build_fare_features,
)
from src.features.fare_features import (
    feature_columns as fare_feature_columns,
)


def test_demand_features_have_lags_and_calendar(clean_trips_df):
    demand = aggregate_hourly_demand(clean_trips_df)
    feats = build_demand_features(demand)

    for col in ("hour", "day_of_week", "is_weekend", "is_holiday", "lag_1"):
        assert col in feats.columns

    # No NaNs remain in the model input columns after warm-up drop.
    model_cols = demand_feature_columns(feats)
    assert feats[model_cols].isna().sum().sum() == 0


def test_demand_lags_use_only_past(clean_trips_df):
    demand = aggregate_hourly_demand(clean_trips_df)
    feats = build_demand_features(demand, dropna=False)

    one_zone = feats[feats["PULocationID"] == feats["PULocationID"].iloc[0]]
    one_zone = one_zone.sort_values("pickup_hour").reset_index(drop=True)
    # lag_1 at row i should equal pickup_count at row i-1.
    assert one_zone["lag_1"].iloc[5] == one_zone["pickup_count"].iloc[4]


def test_fare_features_columns(clean_trips_df):
    feats = build_fare_features(clean_trips_df)

    assert "fare_amount" in feats.columns
    for col in ("trip_distance", "passenger_count", "hour", "is_holiday"):
        assert col in feats.columns

    model_cols = fare_feature_columns(feats)
    assert "fare_amount" not in model_cols
    assert feats[model_cols].isna().sum().sum() == 0
