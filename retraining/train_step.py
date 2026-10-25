"""Train both models inside SageMaker Processing and export the MLflow registry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from scripts.make_mlflow_portable import make_portable


def prepare_registry(seed_dir: Path, output_dir: Path) -> Path:
    """Copy a registry seed into the writable SageMaker output directory."""
    source_db = seed_dir / "mlflow.db"
    source_runs = seed_dir / "mlruns"
    if not source_db.exists() or not source_runs.is_dir():
        raise FileNotFoundError(f"Registry seed must contain mlflow.db and mlruns/: {seed_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_runs = output_dir / "mlruns"
    shutil.copytree(source_runs, output_runs, dirs_exist_ok=True)
    output_db = output_dir / "mlflow.db"
    make_portable(source_db, output_db, f"file://{output_runs.as_posix()}")
    return output_db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--registry-seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=25)
    args = parser.parse_args(argv)

    output_db = prepare_registry(args.registry_seed_dir, args.output_dir)
    os.environ["DATA_FEATURES_DIR"] = str(args.features_dir)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{output_db.as_posix()}"

    from src.models.train import train_all

    summaries = train_all(n_trials=args.n_trials, register=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
