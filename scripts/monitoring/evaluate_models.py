"""Evaluate freshly trained models and emit a quality-gate report.

Used by the retraining pipeline's ``EvaluateModels`` step. Loads each candidate
and seed-registry model, scores them on the same held-out test split, and writes
``evaluation.json`` (consumed by the pipeline ``ConditionStep``)::

    {"demand": {"rmse": ..., "baseline": {...}, "baseline_gate_passed": true}, ...}

Runs locally too::

    python -m scripts.monitoring.evaluate_models --output-dir ./out
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
from copy import deepcopy
from pathlib import Path

from src.config import Config, get_config
from src.models.common import PROBLEMS, load_split, regression_metrics
from src.monitoring.performance import any_degraded, performance_degradation
from src.serving import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate-models")


def prepare_baseline_registry(source_dir: Path, output_dir: Path) -> Path:
    """Copy a mounted seed registry and repoint file-based artifact locations."""
    source_db = source_dir / "mlflow.db"
    source_runs = source_dir / "mlruns"
    if not source_db.exists() or not source_runs.is_dir():
        raise FileNotFoundError(
            f"Baseline registry must contain mlflow.db and mlruns/: {source_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_runs = output_dir / "mlruns"
    shutil.copytree(source_runs, output_runs, dirs_exist_ok=True)
    output_db = output_dir / "mlflow.db"
    shutil.copy2(source_db, output_db)
    artifact_base = f"file://{output_runs.as_posix()}"
    with sqlite3.connect(output_db) as connection:
        targets = (
            ("experiments", "artifact_location"),
            ("runs", "artifact_uri"),
            ("model_versions", "source"),
            ("model_versions", "storage_location"),
        )
        for table, column in targets:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                continue
            connection.execute(
                f"UPDATE {table} SET {column} = ? || "
                f"substr({column}, instr({column}, '/mlruns') + 7) "
                f"WHERE instr({column}, '/mlruns') > 0",
                (artifact_base,),
            )
    return output_db


def evaluate(problem: str, config: Config | None = None) -> dict[str, float]:
    config = config or get_config()
    model = registry.load_model(problem, config)
    X_test, y_test = load_split(config, problem, "test")
    preds = model.predict(X_test)
    metrics = regression_metrics(y_test.to_numpy(), preds)
    logger.info("%s test metrics: %s", problem, metrics)
    return metrics


def _registry_config(config: Config, registry_dir: str) -> Config:
    raw = deepcopy(config.raw)
    database = (Path(registry_dir) / "mlflow.db").resolve()
    raw.setdefault("mlflow", {})["tracking_uri"] = f"sqlite:///{database.as_posix()}"
    return Config(raw=raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate models for the quality gate.")
    parser.add_argument("--output-dir", default="/opt/ml/processing/evaluation")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--baseline-registry-dir", default=None)
    parser.add_argument(
        "--baseline-working-dir", default="/tmp/taxi-forecasting-baseline-registry"
    )
    parser.add_argument("--baseline-tolerance", type=float, default=0.20)
    args = parser.parse_args(argv)

    if args.features_dir:
        os.environ["DATA_FEATURES_DIR"] = args.features_dir
    get_config.cache_clear()
    config = get_config()
    candidate_config = (
        _registry_config(config, args.registry_dir) if args.registry_dir else config
    )
    baseline_config = None
    if args.baseline_registry_dir:
        baseline_working_dir = Path(args.baseline_working_dir)
        prepare_baseline_registry(Path(args.baseline_registry_dir), baseline_working_dir)
        baseline_config = _registry_config(config, str(baseline_working_dir))

    report: dict[str, dict] = {}
    for problem in PROBLEMS:
        candidate = evaluate(problem, candidate_config)
        report[problem] = candidate
        if baseline_config:
            baseline = evaluate(problem, baseline_config)
            comparison = performance_degradation(
                candidate, baseline, tolerance=args.baseline_tolerance
            )
            report[problem].update(
                {
                    "baseline": baseline,
                    "baseline_comparison": comparison,
                    "baseline_gate_passed": not any_degraded(comparison),
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote evaluation report to %s", out_dir / "evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
