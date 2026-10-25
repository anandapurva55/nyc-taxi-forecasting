"""Tests for chronological splitting."""

from __future__ import annotations

import pandas as pd

from src.data.splits import time_based_split


def _frame(n=100):
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x": range(n),
        }
    )


def test_split_sizes_and_order():
    train, val, test = time_based_split(
        _frame(100), "t", train_frac=0.8, val_frac=0.1
    )
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Chronological: train ends before val starts before test starts.
    assert train["t"].max() < val["t"].min()
    assert val["t"].max() < test["t"].min()


def test_split_no_overlap():
    train, val, test = time_based_split(
        _frame(50), "t", train_frac=0.6, val_frac=0.2
    )
    all_x = set(train["x"]) | set(val["x"]) | set(test["x"])
    assert len(all_x) == 50
