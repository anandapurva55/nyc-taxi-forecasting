"""Generate a feature-drift report locally for the dashboard (no AWS).

This is a convenience wrapper around :func:`src.monitoring.compute_drift` that
produces the same ``<problem>_drift_metrics.json`` schema the scheduled
SageMaker drift job (``scripts/monitoring/run_drift_job.py``) writes to S3 in
production. It lets the dashboard's Monitoring section show a live drift panel
without any AWS resources.

It samples the model's training split as the *reference*, derives a *current*
sample that is either drift-free (``--mode clean``) or deliberately shifted
(``--mode drift``), computes PSI + KS drift, and writes the report to
``artifacts/drift/<problem>_drift_metrics.json`` (read by the dashboard).

Examples::

    python -m scripts.monitoring.demo_drift_report --problem fare --mode drift
    python -m scripts.monitoring.demo_drift_report --problem demand --mode clean
    python -m scripts.monitoring.demo_drift_report --problem all --mode drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.models.common import PROBLEMS, load_split
from src.monitoring import compute_drift

DEFAULT_OUTPUT_DIR = Path("artifacts/drift")


def _make_current(reference: pd.DataFrame, mode: str, rng: np.random.Generator) -> pd.DataFrame:
    """A sample of the reference, optionally shifted to induce drift."""
    n = min(1500, len(reference))
    current = reference.sample(n=n, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)
    if mode == "clean":
        return current

    # Shift the two highest-variance numeric columns so the demo crosses the PSI threshold.
    numeric = current.select_dtypes(include="number")
    spread = numeric.std().sort_values(ascending=False)
    for col in spread.index[:2]:
        current[col] = current[col] * 2.5 + 2.0 * spread[col]
    return current


def build_report(problem: str, mode: str, rng: np.random.Generator) -> dict:
    config = get_config()
    reference, _ = load_split(config, problem, "train")
    current = _make_current(reference, mode, rng)
    report = compute_drift(reference, current)
    return {"problem": problem, "n_current": int(len(current)), **report.to_dict()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a local drift report for the dashboard.")
    parser.add_argument(
        "--problem", choices=[*PROBLEMS, "all"], default="fare", help="Model to report on."
    )
    parser.add_argument(
        "--mode", choices=["drift", "clean"], default="drift", help="Induce drift or not."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = list(PROBLEMS) if args.problem == "all" else [args.problem]
    for problem in problems:
        summary = build_report(problem, args.mode, rng)
        out_file = output_dir / f"{problem}_drift_metrics.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"{problem}: drifted={summary['drifted']} "
            f"({summary['n_drifted']}/{summary['n_features']}) -> {out_file}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
