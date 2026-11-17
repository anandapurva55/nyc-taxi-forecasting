"""Unit tests for monitoring (drift, performance, capture)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from scripts.monitoring.run_drift_job import (
    _align_capture_columns,
    _cloudwatch_metric_data,
    build_parser,
)
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


def test_align_capture_columns_matches_training_feature_names():
    fare = pd.DataFrame({"pu_location_id": [132], "do_location_id": [230], "trip_distance": [3.4]})
    demand = pd.DataFrame({"zone": [132], "hour_of_day": [18], "day_of_week": [1]})

    assert list(_align_capture_columns(fare, "fare").columns) == [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
    ]
    assert list(_align_capture_columns(demand, "demand").columns) == [
        "PULocationID",
        "hour",
        "day_of_week",
    ]


def test_fail_on_drift_defaults_on_and_can_be_disabled():
    assert build_parser().parse_args(["--problem", "fare"]).fail_on_drift is True
    assert build_parser().parse_args(["--problem", "fare", "--no-fail-on-drift"]).fail_on_drift is False


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


def test_zone_ids_are_compared_as_categories():
    rng = np.random.default_rng(7)
    zones = rng.choice([132, 161, 230, 236, 237], size=3000, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    ref = pd.DataFrame({"PULocationID": zones})
    same = pd.DataFrame({"PULocationID": rng.permutation(zones)[:1000]})
    shifted = pd.DataFrame({"PULocationID": np.full(1000, 132)})
    stable = compute_drift(ref, same).features[0]
    moved = compute_drift(ref, shifted).features[0]
    assert stable.kind == "categorical" and stable.test == "chi2"
    assert not stable.drifted
    assert moved.drifted and moved.pvalue < 0.01


def test_prediction_drift_flags_shifted_outputs():
    from src.monitoring.drift import compute_prediction_drift

    rng = np.random.default_rng(8)
    target = pd.Series(rng.normal(15, 5, 5000))
    assert not compute_prediction_drift(target, pd.Series(rng.normal(15, 5, 800))).drifted
    assert compute_prediction_drift(target, pd.Series(rng.normal(40, 5, 800))).drifted


# --- Calendar features --------------------------------------------------------
def _hourly_reference(days: int = 14) -> pd.DataFrame:
    hours = pd.date_range("2024-02-05", periods=24 * days, freq="h")
    return pd.DataFrame({"hour": hours.hour, "day_of_week": hours.dayofweek})


def _traffic(start: str, end: str, hours_allowed=None, n: int = 600, seed: int = 0) -> pd.DataFrame:
    span = pd.date_range(start, end, freq="h")
    if hours_allowed is not None:
        span = span[span.hour.isin(hours_allowed)]
    picks = pd.DatetimeIndex(np.random.default_rng(seed).choice(span, n))
    return pd.DataFrame({"hour": picks.hour, "day_of_week": picks.dayofweek})


def _flags(report):
    return {f.feature: f.drifted for f in report.features}


def test_short_capture_window_is_not_calendar_drift():
    from src.monitoring.drift import compute_drift

    window = (pd.Timestamp("2024-02-27 19:00"), pd.Timestamp("2024-02-29 23:00"))
    report = compute_drift(_hourly_reference(), _traffic(*window), window=window)
    assert _flags(report) == {"hour": False, "day_of_week": False}
    assert {f.kind for f in report.features} == {"calendar"}


def test_traffic_piled_onto_some_hours_of_the_window_is_flagged():
    from src.monitoring.drift import compute_drift

    window = (pd.Timestamp("2024-02-27 00:00"), pd.Timestamp("2024-02-29 23:00"))
    nights = _traffic(*window, hours_allowed=[0, 1, 2, 3, 22, 23])
    assert _flags(compute_drift(_hourly_reference(), nights, window=window))["hour"] is True


def test_reference_cells_are_weighted_equally():
    from src.monitoring.drift import window_matched

    ref = _hourly_reference(days=14)
    window = (pd.Timestamp("2024-02-27 19:00"), pd.Timestamp("2024-02-29 23:00"))
    matched = window_matched(ref, ref, ["hour", "day_of_week"], window)
    # 5 hours on Tuesday plus two full days, one row per cell.
    assert len(matched) == 5 + 24 + 24
    assert matched["day_of_week"].value_counts().to_dict() == {1: 5, 2: 24, 3: 24}


def test_capture_window_comes_from_target_hours():
    from scripts.monitoring.run_drift_job import _capture_window

    frame = pd.DataFrame({"target_hour": ["2024-02-28T05:00:00", "2024-02-27T19:00:00", None]})
    assert _capture_window(frame) == (pd.Timestamp("2024-02-27 19:00"), pd.Timestamp("2024-02-28 05:00"))
    assert _capture_window(pd.DataFrame({"zone": [1]})) is None


def test_category_noise_in_a_small_window_is_not_drift():
    from src.monitoring.drift import compute_drift

    rng = np.random.default_rng(3)
    zones = np.arange(100)
    reference = pd.DataFrame({"PULocationID": rng.choice(zones, 50_000)})
    report = compute_drift(reference, pd.DataFrame({"PULocationID": rng.choice(zones, 400)}))
    feature = report.features[0]
    # Sampling noise alone pushes category PSI near the threshold, but it is not significant.
    assert feature.pvalue > 0.05 and not feature.drifted


def test_real_category_shift_is_flagged():
    from src.monitoring.drift import compute_drift

    rng = np.random.default_rng(4)
    zones = np.arange(100)
    reference = pd.DataFrame({"PULocationID": rng.choice(zones, 50_000)})
    current = rng.choice(zones, 400)
    current[:120] = 7  # 30% of requests now come from one zone
    assert compute_drift(reference, pd.DataFrame({"PULocationID": current})).features[0].drifted

