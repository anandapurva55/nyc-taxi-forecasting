"""Unit tests for live demand-performance monitoring."""

from __future__ import annotations

import json

import pandas as pd

from src.monitoring.live_performance import (
    cloudwatch_metrics,
    evaluate_live_demand,
    join_demand_outcomes,
)

ACTUALS = pd.DataFrame(
    {
        "PULocationID": [132, 132, 230],
        "pickup_hour": pd.to_datetime(["2024-02-29 20:00", "2024-02-29 21:00", "2024-02-29 20:00"]),
        "pickup_count": [100, 80, 40],
    }
)


def _captured(predictions):
    return pd.DataFrame(
        [
            {"zone": z, "target_hour": h, "prediction": p, "timestamp": f"2026-11-17T10:00:0{i}"}
            for i, (z, h, p) in enumerate(predictions)
        ]
    )


def test_join_matches_zone_and_hour_and_keeps_latest_duplicate():
    captured = _captured(
        [
            (132, "2024-02-29T20:00:00", 90.0),
            (132, "2024-02-29T20:15:00", 95.0),  # same hour, later: wins
            (230, "2024-02-29T20:00:00", 42.0),
            (161, "2024-02-29T20:00:00", 10.0),  # no actuals for this zone
        ]
    )
    joined = join_demand_outcomes(captured, ACTUALS)
    assert len(joined) == 2
    assert joined.set_index("zone").loc[132, "prediction"] == 95.0


def test_evaluate_flags_degradation_against_baseline():
    good = evaluate_live_demand(_captured([(132, "2024-02-29T20:00", 98.0)]), ACTUALS, {"rmse": 5.0})
    bad = evaluate_live_demand(_captured([(132, "2024-02-29T20:00", 40.0)]), ACTUALS, {"rmse": 5.0})
    assert good["matched"] == 1 and not good["degraded"]
    assert bad["degraded"] and bad["degradation"]["rmse"]["degraded"]
    names = {m["MetricName"] for m in cloudwatch_metrics(bad)}
    assert {"PerformanceDegraded", "MatchedPredictions", "LiveRMSE", "LiveWAPE"} <= names


def test_evaluate_reports_no_actuals_without_flagging():
    summary = evaluate_live_demand(_captured([(1, "2030-01-01T00:00", 3.0)]), ACTUALS, {"rmse": 5.0})
    assert summary["status"] == "no_matched_actuals" and summary["degraded"] is False


def test_performance_job_runs_end_to_end_locally(tmp_path):
    from scripts.monitoring.run_performance_job import main

    capture_dir = tmp_path / "captured"
    capture_dir.mkdir()
    line = {
        "event": "prediction",
        "problem": "demand",
        "features": {"zone": 132, "target_hour": "2024-02-29T20:00:00"},
        "prediction": 97.0,
    }
    (capture_dir / "part-0").write_text(json.dumps(line) + "\n")
    actuals = tmp_path / "actuals.parquet"
    ACTUALS.to_parquet(actuals)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"demand": {"test_metrics": {"rmse": 5.0, "mae": 3.0}}}))
    out = tmp_path / "out"
    code = main([
        "--current-dir", str(capture_dir), "--actuals", str(actuals), "--baseline", str(baseline),
        "--output-dir", str(out), "--no-cloudwatch",
    ])
    result = json.loads((out / "performance_metrics.json").read_text())
    assert code == 0 and result["matched"] == 1 and result["degraded"] is False
