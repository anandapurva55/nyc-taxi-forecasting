"""SageMaker Processing job entrypoint: live accuracy of served demand predictions.

Runs on a schedule (EventBridge -> SageMaker Processing). It:

1. Loads captured demand predictions (Firehose output in S3, or a local folder).
2. Loads realised hourly pickup counts (the demand table built by the pipeline
   from the newest published TLC file), from S3 or a local path.
3. Joins them on zone and target hour and computes live RMSE, MAE and WAPE.
4. Compares live error with the served model's test baseline
   (``config/performance_baseline.json``, written at training time).
5. Writes ``performance_metrics.json`` and publishes CloudWatch metrics, including
   ``PerformanceDegraded`` which drives an alarm.

Local dry-run (no AWS)::

    python -m scripts.monitoring.run_performance_job --current-dir ./captured \\
        --actuals data/processed/demand_hourly.parquet --output-dir ./out --no-cloudwatch
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.monitoring.capture import load_capture_file, load_capture_s3
from src.monitoring.live_performance import cloudwatch_metrics, evaluate_live_demand
from src.monitoring.performance import DEFAULT_TOLERANCE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("performance-job")

BASELINE_PATH = PROJECT_ROOT / "config" / "performance_baseline.json"


def _load_current(current_dir: str | None, current_s3: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if current_dir:
        for file in sorted(Path(current_dir).glob("*")):
            if file.is_file():
                frames.append(load_capture_file(file, problem="demand"))
    if current_s3:
        bucket, _, prefix = current_s3.replace("s3://", "").partition("/")
        frames.append(load_capture_s3(bucket, prefix, problem="demand"))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_actuals(location: str) -> pd.DataFrame:
    if location.startswith("s3://"):  # pragma: no cover - requires AWS
        import boto3

        bucket, _, key = location.replace("s3://", "").partition("/")
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    return pd.read_parquet(location)


def _load_baseline(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data["demand"]["test_metrics"].items()}


def _push_cloudwatch(namespace: str, metrics: list[dict]) -> None:  # pragma: no cover
    import boto3

    boto3.client("cloudwatch").put_metric_data(Namespace=namespace, MetricData=metrics)
    logger.info("Published %d metrics to CloudWatch namespace %s", len(metrics), namespace)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captured = _load_current(args.current_dir, args.current_s3)
    actuals = _load_actuals(args.actuals)
    baseline = _load_baseline(Path(args.baseline))
    summary = evaluate_live_demand(captured, actuals, baseline, tolerance=args.tolerance)
    (output_dir / "performance_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "Live demand performance: matched %d of %d predictions, degraded=%s, current=%s",
        summary["matched"], summary["captured"], summary["degraded"], summary.get("current"),
    )
    if not args.no_cloudwatch:
        _push_cloudwatch(args.namespace, cloudwatch_metrics(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live demand-prediction accuracy job.")
    p.add_argument("--current-dir", default=None, help="Local folder of captured predictions.")
    p.add_argument("--current-s3", default=None, help="s3://bucket/prefix of captured predictions.")
    p.add_argument("--actuals", required=True, help="Hourly demand parquet (local path or s3://).")
    p.add_argument("--baseline", default=str(BASELINE_PATH), help="Baseline metrics JSON.")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--output-dir", default="/opt/ml/processing/output")
    p.add_argument("--namespace", default="TaxiForecasting/Performance")
    p.add_argument("--no-cloudwatch", action="store_true", help="Skip CloudWatch publish.")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
