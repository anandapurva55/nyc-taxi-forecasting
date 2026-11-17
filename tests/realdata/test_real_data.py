"""Checks on the real TLC data and the pipeline outputs built from it.

The unit and integration tests use synthetic data, so they cannot catch a
schema change in a newly published TLC file or a leak in the real splits. These
tests run against data/ when the pipeline has been run, and skip otherwise (for
example in CI, where the data is not downloaded).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import get_config
from src.data.validation import REQUIRED_COLUMNS

CONFIG = get_config()
FEATURES = CONFIG.features_dir
PROCESSED = CONFIG.processed_dir

pytestmark = pytest.mark.skipif(
    not (FEATURES / "demand_test.parquet").exists(), reason="pipeline outputs not present"
)


def _raw_file():
    year, month = int(CONFIG.data["tlc_year"]), int(CONFIG.data["tlc_month"])
    path = CONFIG.raw_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if not path.exists():
        pytest.skip(f"raw file not present: {path.name}")
    return path


def test_raw_file_has_the_expected_schema():
    columns = set(pd.read_parquet(_raw_file()).columns)
    assert set(REQUIRED_COLUMNS) <= columns


@pytest.mark.parametrize(("problem", "time_col"), [("demand", "pickup_hour"), ("fare", "tpep_pickup_datetime")])
def test_splits_are_chronological_and_disjoint(problem, time_col):
    parts = {s: pd.read_parquet(FEATURES / f"{problem}_{s}.parquet", columns=[time_col])[time_col]
             for s in ("train", "val", "test")}
    assert parts["train"].max() <= parts["val"].min()
    assert parts["val"].max() <= parts["test"].min()
    for part in parts.values():
        assert part.is_monotonic_increasing


def test_demand_lags_only_use_past_hours():
    features = pd.read_parquet(FEATURES / "demand_test.parquet")
    hourly = pd.read_parquet(PROCESSED / "demand_hourly.parquet").set_index(["PULocationID", "pickup_hour"])
    sample = features.sample(500, random_state=0)
    for lag in (1, 24, 168):
        keys = list(zip(sample["PULocationID"], sample["pickup_hour"] - pd.Timedelta(hours=lag), strict=True))
        expected = hourly["pickup_count"].reindex(keys).to_numpy()
        assert (sample[f"lag_{lag}"].to_numpy() == expected).all(), f"lag_{lag} does not match the count {lag}h earlier"


def test_demand_counts_include_trips_the_fare_rules_drop():
    demand_total = pd.read_parquet(PROCESSED / "demand_hourly.parquet", columns=["pickup_count"])["pickup_count"].sum()
    clean_trips = len(pd.read_parquet(PROCESSED / "trips_clean.parquet", columns=["PULocationID"]))
    assert demand_total > clean_trips
