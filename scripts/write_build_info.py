"""Write build_info.json: the provenance record baked into every API image.

Ties a deployed image to the Git commit it was built from, the exact registered
model versions (and their MLflow run IDs) and a SHA-256 fingerprint of every
baked input file, so any running service can be traced and rolled back.

Usage::

    python scripts/write_build_info.py --commit "$(git rev-parse HEAD)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAKED_INPUTS = ["mlflow.db", "data/processed/demand_hourly.parquet"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_models(db_path: Path) -> list[dict]:
    """Latest registered version of every model, with its source run."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name, MAX(CAST(version AS INTEGER)) FROM model_versions GROUP BY name"
        ).fetchall()
        models = []
        for name, version in rows:
            run_id = conn.execute(
                "SELECT run_id FROM model_versions WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()[0]
            models.append({"name": name, "version": int(version), "run_id": run_id})
    return sorted(models, key=lambda m: m["name"])


def build_info(commit: str, root: Path = ROOT) -> dict:
    return {
        "git_commit": commit,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": registered_models(root / "mlflow.db"),
        "inputs": {
            name: sha256(root / name) for name in BAKED_INPUTS if (root / name).exists()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--output", default=str(ROOT / "config" / "build_info.json"))
    args = parser.parse_args()
    info = build_info(args.commit)
    Path(args.output).write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
