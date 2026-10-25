"""Batch entry point: forecast the next N hours of demand to SQLite.

Usage::

    python -m scripts.batch_forecast                 # next 24h (config default)
    python -m scripts.batch_forecast --horizon 12    # next 12h
"""

from __future__ import annotations

import argparse
import json
import logging

from src.serving.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the batch demand forecast.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Hours ahead to forecast (default: config serving.batch_horizon_hours).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    summary = run_batch(horizon_hours=args.horizon)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
