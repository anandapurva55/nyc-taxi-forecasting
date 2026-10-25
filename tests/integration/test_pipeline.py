"""Integration test: end-to-end pipeline on synthetic data.

Exercises clean -> aggregate -> features -> split without touching the network,
writing artifacts into a temporary directory.
"""

from __future__ import annotations

from src.data.aggregation import aggregate_hourly_demand
from src.data.splits import time_based_split
from src.data.validation import clean_trips
from src.features.demand_features import build_demand_features
from src.features.fare_features import build_fare_features


def test_pipeline_end_to_end(raw_trips, tmp_path):
    clean, report = clean_trips(raw_trips)
    assert report.output_rows > 0

    demand = aggregate_hourly_demand(clean)
    demand_feats = build_demand_features(demand)
    fare_feats = build_fare_features(clean)

    assert len(demand_feats) > 0
    assert len(fare_feats) > 0

    d_train, d_val, d_test = time_based_split(demand_feats, "pickup_hour")
    f_train, f_val, f_test = time_based_split(fare_feats, "tpep_pickup_datetime")

    assert len(d_train) + len(d_val) + len(d_test) == len(demand_feats)
    assert len(f_train) + len(f_val) + len(f_test) == len(fare_feats)

    # Round-trip through parquet to confirm artifacts are serialisable.
    out = tmp_path / "demand_features.parquet"
    demand_feats.to_parquet(out, index=False)
    assert out.exists()
