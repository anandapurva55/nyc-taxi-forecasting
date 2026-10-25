"""SageMaker Processing job entrypoint: feature-drift detection.

Runs on a schedule (EventBridge -> SageMaker Processing). It:

1. Loads the **reference** feature distribution - the training split for each
   problem (baked reference parquet, or read from ``--reference-dir``).
2. Loads the **current** captured inference inputs - prediction-capture records
   shipped to S3 by Firehose (``--current-s3`` or a local ``--current-dir``).
3. Computes per-feature PSI + KS drift via :func:`src.monitoring.compute_drift`.
4. Writes ``drift_metrics.json`` (and an optional Evidently HTML report) to the
   output directory (SageMaker uploads it to S3).
5. Pushes summary metrics to CloudWatch (best-effort).
6. Exits non-zero when drift is detected so the Pipeline/alarm can react.

Local dry-run (no AWS)::

    python -m scripts.monitoring.run_drift_job --problem fare \
        --current-dir ./captured --output-dir ./out --no-cloudwatch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import get_config
from src.models.common import PROBLEMS, load_split
from src.monitoring import compute_drift
from src.monitoring.capture import load_capture_file, load_capture_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("drift-job")


def _load_reference(problem: str, reference_dir: str | None) -> pd.DataFrame:
    """Reference = training-split feature matrix for the problem."""
    if reference_dir:
        path = Path(reference_dir) / f"{problem}_reference.parquet"
        logger.info("Loading reference from %s", path)
        return pd.read_parquet(path)
    config = get_config()
    X, _ = load_split(config, problem, "train")
    return X


def _load_current(
    problem: str, current_dir: str | None, current_s3: str | None
) -> pd.DataFrame:
    """Current = captured inference inputs for the problem."""
    frames: list[pd.DataFrame] = []
    if current_dir:
        for file in sorted(Path(current_dir).glob("*")):
            if file.is_file():
                frames.append(load_capture_file(file, problem=problem))
    if current_s3:
        bucket, _, prefix = current_s3.replace("s3://", "").partition("/")
        frames.append(load_capture_s3(bucket, prefix, problem=problem))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return _align_capture_columns(pd.concat(frames, ignore_index=True), problem)


# Captured request fields use API names; the reference uses training feature names.
CAPTURE_TO_FEATURE = {
    "fare": {"pu_location_id": "PULocationID", "do_location_id": "DOLocationID"},
    "demand": {"zone": "PULocationID", "hour_of_day": "hour"},
}


def _align_capture_columns(current: pd.DataFrame, problem: str) -> pd.DataFrame:
    """Rename captured API fields so they line up with the reference features."""
    return current.rename(columns=CAPTURE_TO_FEATURE.get(problem, {}))


def _cloudwatch_metric_data(metrics: list[dict], problem: str) -> list[dict]:
    """Return aggregate alarm metrics plus per-problem diagnostic metrics."""
    aggregate = [dict(metric) for metric in metrics]
    dimensional = [
        {**metric, "Dimensions": [{"Name": "Problem", "Value": problem}]}
        for metric in metrics
    ]
    return aggregate + dimensional


def _push_cloudwatch(namespace: str, metrics: list[dict], problem: str) -> None:  # pragma: no cover
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not available; skipping CloudWatch publish.")
        return
    cw = boto3.client("cloudwatch")
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=_cloudwatch_metric_data(metrics, problem),
    )
    logger.info("Published %d metrics to CloudWatch namespace %s", len(metrics) * 2, namespace)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = _load_reference(args.problem, args.reference_dir)
    current = _load_current(args.problem, args.current_dir, args.current_s3)

    if current.empty:
        logger.warning("No captured records for problem=%s; nothing to evaluate.", args.problem)
        (output_dir / "drift_metrics.json").write_text(
            json.dumps({"problem": args.problem, "status": "no_data", "drifted": False}, indent=2)
        )
        return 0

    report = compute_drift(
        reference, current, psi_threshold=args.psi_threshold, ks_alpha=args.ks_alpha
    )
    summary = {"problem": args.problem, "n_current": int(len(current)), **report.to_dict()}
    (output_dir / "drift_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Drift: %d/%d features drifted (share=%.2f) -> drifted=%s",
        report.n_drifted,
        report.n_features,
        report.share_drifted,
        report.drifted,
    )

    if args.evidently:  # pragma: no cover - optional heavy dependency
        try:
            from src.monitoring.drift import build_evidently_report

            common = [c for c in reference.columns if c in current.columns]
            html = build_evidently_report(reference[common], current[common])
            html.save_html(str(output_dir / f"{args.problem}_drift_report.html"))
        except Exception as exc:
            logger.warning("Evidently report skipped: %s", exc)

    if not args.no_cloudwatch:  # pragma: no cover - requires AWS
        _push_cloudwatch(args.namespace, report.cloudwatch_metrics(), args.problem)

    return 1 if (report.drifted and args.fail_on_drift) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feature-drift detection job.")
    p.add_argument("--problem", choices=list(PROBLEMS), required=True)
    p.add_argument("--reference-dir", default=None, help="Dir with <problem>_reference.parquet.")
    p.add_argument("--current-dir", default=None, help="Local dir of captured JSON-line files.")
    p.add_argument("--current-s3", default=None, help="s3://bucket/prefix of captured records.")
    p.add_argument("--output-dir", default="/opt/ml/processing/output")
    p.add_argument("--psi-threshold", type=float, default=0.2)
    p.add_argument("--ks-alpha", type=float, default=0.05)
    p.add_argument("--namespace", default="TaxiForecasting/Drift")
    p.add_argument("--evidently", action="store_true", help="Also render an Evidently HTML report.")
    p.add_argument("--no-cloudwatch", action="store_true", help="Skip CloudWatch publish.")
    p.add_argument(
        "--fail-on-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when drift is detected (default). Use --no-fail-on-drift "
        "to finish successfully and signal drift only through CloudWatch.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
