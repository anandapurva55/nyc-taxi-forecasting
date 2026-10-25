"""Rewrite MLflow artifact paths so a baked SQLite registry works inside a
Linux container.

The dev ``mlflow.db`` records **absolute host paths**, e.g.
``file:///C:/taxi-forecasting/mlruns/1/<run>/artifacts/model``. Those cannot be
resolved when the database and ``mlruns/`` tree are copied into an image at
``/app``. This script produces a *container-ready* copy of the database with
every artifact URI repointed at ``<new_base>`` (default ``file:///app/mlruns``).

It updates every ``.../mlruns`` prefixed value across the tables/columns that
MLflow uses to locate artifacts, and is idempotent (safe to run repeatedly).

Usage:
    python scripts/make_mlflow_portable.py \
        --src mlflow.db --dest mlflow.container.db --new-base file:///app/mlruns
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
from pathlib import Path

# (table, column) pairs MLflow uses to store artifact locations. Columns that do
# not exist in a given schema version are skipped automatically.
_TARGETS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("model_versions", "source"),
    ("model_versions", "storage_location"),
]

# Matches everything up to and including the ``/mlruns`` segment of a file URI,
# e.g. ``file:///C:/taxi-forecasting/mlruns`` or ``file:///app/mlruns``.
_PREFIX_RE = re.compile(r"^file://.*?/mlruns", re.IGNORECASE)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    return column in cols


def repoint(conn: sqlite3.Connection, new_base: str) -> int:
    """Repoint all mlruns-based artifact URIs to ``new_base``. Returns row count."""
    changed = 0
    for table, column in _TARGETS:
        if not _column_exists(conn, table, column):
            continue
        rows = conn.execute(
            f"SELECT rowid, {column} FROM {table} WHERE {column} LIKE 'file:%mlruns%'"
        ).fetchall()
        for rowid, value in rows:
            new_value = _PREFIX_RE.sub(new_base, value)
            if new_value != value:
                conn.execute(
                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                    (new_value, rowid),
                )
                changed += 1
    conn.commit()
    return changed


def make_portable(src: Path, dest: Path, new_base: str) -> int:
    if not src.exists():
        raise FileNotFoundError(f"Source MLflow DB not found: {src}")
    shutil.copyfile(src, dest)
    with sqlite3.connect(dest) as conn:
        return repoint(conn, new_base.rstrip("/"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="mlflow.db", type=Path)
    parser.add_argument("--dest", default="mlflow.container.db", type=Path)
    parser.add_argument("--new-base", default="file:///app/mlruns")
    args = parser.parse_args()

    changed = make_portable(args.src, args.dest, args.new_base)
    print(f"Wrote {args.dest} ({changed} artifact path(s) repointed to {args.new_base}).")


if __name__ == "__main__":
    main()
