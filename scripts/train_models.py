"""Model training entry point: train, tune, evaluate and register the tree models.

Usage::

    python -m scripts.train_models                      # all problems, config defaults
    python -m scripts.train_models --problem demand     # one problem
    python -m scripts.train_models --n-trials 10        # faster tuning
    python -m scripts.train_models --no-register        # skip Model Registry
"""

from __future__ import annotations

import argparse
import json
import logging

from src.config import PROJECT_ROOT
from src.models.common import PROBLEMS
from src.models.train import train_all

BASELINE_PATH = PROJECT_ROOT / "config" / "performance_baseline.json"

logger = logging.getLogger(__name__)


def write_performance_baseline(summaries: dict, path=BASELINE_PATH) -> None:
    """Record each registered champion's test metrics for live-performance checks.

    Existing entries for problems not retrained in this run are kept.
    """
    baseline = json.loads(path.read_text()) if path.exists() else {}
    for problem, summary in summaries.items():
        if not summary.get("registered"):
            continue
        baseline[problem] = {
            "model": summary["registered"],
            "test_metrics": summary["final"]["test_metrics"],
            "test_rmse_ci95": summary["final"]["test_metrics_ci95"]["rmse"],
        }
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    logger.info("Wrote performance baseline to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model training.")
    parser.add_argument(
        "--problem",
        choices=[*PROBLEMS, "all"],
        default="all",
        help="Which problem to train (default: all).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optuna trials per estimator (default: config models.n_trials).",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Train and log runs but do not register to the MLflow Model Registry.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    problems = None if args.problem == "all" else [args.problem]
    summaries = train_all(
        problems=problems, n_trials=args.n_trials, register=not args.no_register
    )
    print(json.dumps(summaries, indent=2, default=str))
    if not args.no_register:
        write_performance_baseline(summaries)


if __name__ == "__main__":
    main()
