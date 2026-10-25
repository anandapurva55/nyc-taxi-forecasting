"""Unit tests for monitoring (drift, performance, capture)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from scripts.monitoring.run_drift_job import _cloudwatch_metric_data
from src.monitoring import (
    compute_drift,
    performance_degradation,
    population_stability_index,
)
from src.monitoring.capture import _capture_lines_from_bytes, parse_capture_lines
from src.monitoring.performance import any_degraded, evaluate_performance


def _frame(rng, loc, n=2000):
    return pd.DataFrame(
        {
            "trip_distance": rng.normal(loc, 1.0, n),
            "passenger_count": rng.integers(1, 4, n),
        }
    )


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert population_stability_index(x, x) < 1e-6


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(5, 1, 5000)
    assert population_stability_index(ref, cur) > 0.2


def test_compute_drift_no_drift():
    rng = np.random.default_rng(2)
    ref = _frame(rng, 3.0)
    cur = _frame(rng, 3.0)
    report = compute_drift(ref, cur)
    assert report.n_features == 2
    assert not report.drifted
    assert report.share_drifted == 0.0


def test_compute_drift_detects_strong_drift():
    rng = np.random.default_rng(3)
    ref = _frame(rng, 3.0)
    cur = _frame(rng, 12.0)
    report = compute_drift(ref, cur)
    assert report.drifted
    assert any(f.feature == "trip_distance" and f.drifted for f in report.features)
    # Summary shapes into CloudWatch metric dicts.
    metrics = report.cloudwatch_metrics()
    assert {m["MetricName"] for m in metrics} == {"FeaturesDrifted", "ShareDrifted", "MaxPSI"}


def test_cloudwatch_metric_data_includes_aggregate_and_problem_series():
    metrics = [{"MetricName": "FeaturesDrifted", "Value": 2.0, "Unit": "Count"}]

    metric_data = _cloudwatch_metric_data(metrics, "fare")

    assert metric_data == [
        metrics[0],
        {
            **metrics[0],
            "Dimensions": [{"Name": "Problem", "Value": "fare"}],
        },
    ]


def test_evaluate_performance_and_degradation():
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    y_pred = np.array([10.5, 11.5, 14.5, 15.5])
    metrics = evaluate_performance(y_true, y_pred)
    assert set(metrics) == {"rmse", "mae", "mape", "r2"}

    baseline = {"rmse": 0.5, "mae": 0.4, "mape": 4.0, "r2": 0.95}
    # Current is much worse on error metrics.
    current = {"rmse": 2.0, "mae": 1.5, "mape": 20.0, "r2": 0.4}
    deg = performance_degradation(current, baseline)
    assert deg["rmse"]["degraded"] is True
    assert deg["r2"]["degraded"] is True
    assert any_degraded(deg)


def test_degradation_not_flagged_within_tolerance():
    baseline = {"rmse": 1.0, "r2": 0.9}
    current = {"rmse": 1.1, "r2": 0.88}  # small changes
    deg = performance_degradation(current, baseline, tolerance=0.2)
    assert not any_degraded(deg)


def test_degradation_handles_zero_baseline():
    deg = performance_degradation({"rmse": 1.0}, {"rmse": 0.0})
    assert deg["rmse"]["degraded"] is False
    assert deg["rmse"]["rel_change"] is None


def test_parse_capture_lines_flattens_features():
    lines = [
        json.dumps(
            {
                "event": "prediction",
                "problem": "fare",
                "model_version": "2",
                "features": {"trip_distance": 3.1, "passenger_count": 1},
                "prediction": 14.2,
            }
        ),
        json.dumps({"event": "request", "path": "/health"}),  # ignored
        "not json",  # ignored
        json.dumps(
            {
                "event": "prediction",
                "problem": "demand",
                "features": {"zone": 100, "hour_of_day": 8},
                "prediction": 42.0,
            }
        ),
    ]
    df = parse_capture_lines(lines)
    assert len(df) == 2
    assert "trip_distance" in df.columns
    assert "prediction" in df.columns

    fare = parse_capture_lines(lines, problem="fare")
    assert len(fare) == 1
    assert fare.iloc[0]["prediction"] == 14.2


def test_parse_double_gzip_cloudwatch_capture():
    prediction = {
        "event": "prediction",
        "problem": "fare",
        "features": {"trip_distance": 3.1},
        "prediction": 14.2,
    }
    envelope = {
        "messageType": "DATA_MESSAGE",
        "logEvents": [{"message": json.dumps(prediction)}],
    }
    second_envelope = {
        "messageType": "DATA_MESSAGE",
        "logEvents": [{"message": json.dumps({**prediction, "prediction": 15.0})}],
    }
    cloudwatch_batches = gzip.compress(json.dumps(envelope).encode()) + gzip.compress(
        json.dumps(second_envelope).encode()
    )
    payload = gzip.compress(cloudwatch_batches)

    frame = parse_capture_lines(_capture_lines_from_bytes(payload))

    assert len(frame) == 2
    assert frame.iloc[0]["trip_distance"] == 3.1
    assert frame.iloc[0]["prediction"] == 14.2
    assert frame.iloc[1]["prediction"] == 15.0
