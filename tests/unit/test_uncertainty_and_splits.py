"""Unit tests for bootstrap intervals, the tuning holdout and demand pickup selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import select_pickups
from src.models.common import chronological_holdout, rmse
from src.models.uncertainty import block_bootstrap_ci, paired_difference_ci


def test_block_bootstrap_ci_brackets_observed_rmse():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 2, 600)
    pred = y + rng.normal(0, 1, 600)
    blocks = np.repeat(np.arange(60), 10)
    low, high = block_bootstrap_ci(y, pred, blocks, rmse, n_boot=200)
    assert low < rmse(y, pred) < high


def test_paired_difference_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(10, 2, 600)
    good = y + rng.normal(0, 0.5, 600)
    bad = y + rng.normal(0, 3.0, 600)
    blocks = np.repeat(np.arange(60), 10)
    observed, low, high = paired_difference_ci(y, good, bad, blocks, rmse, n_boot=200)
    assert observed < 0 and high < 0


def test_paired_difference_interval_contains_zero_for_equal_models():
    rng = np.random.default_rng(2)
    y = rng.normal(10, 2, 600)
    a = y + rng.normal(0, 1, 600)
    b = y + rng.normal(0, 1, 600)
    blocks = np.repeat(np.arange(60), 10)
    _, low, high = paired_difference_ci(y, a, b, blocks, rmse, n_boot=200)
    assert low < 0 < high


def test_chronological_holdout_takes_the_latest_rows():
    X = pd.DataFrame({"t": range(100)})
    y = pd.Series(range(100))
    X_fit, y_fit, X_hold, y_hold = chronological_holdout(X, y, 0.2)
    assert len(X_fit) == 80 and len(X_hold) == 20
    assert X_hold["t"].min() > X_fit["t"].max()
    with pytest.raises(ValueError):
        chronological_holdout(X, y, 1.0)


def test_select_pickups_keeps_trips_the_fare_rules_would_drop():
    raw = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2024-02-03 08:10", "2024-02-03 08:20", "2024-03-01 00:05", None]
            ),
            "PULocationID": [132, 132, 132, 48],
            # These would fail clean_trips but are still real pickups.
            "passenger_count": [0, None, 1, 1],
            "fare_amount": [-5.0, 900.0, 10.0, 10.0],
        }
    )
    kept, report = select_pickups(raw, year=2024, month=2)
    assert len(kept) == 2
    assert report.dropped == {"null_pickup_time_or_zone": 1, "outside_target_period": 1}
