"""Taxi demand & fare prediction ML platform."""

import os

# MLflow 3 logs a tooling hint on import; keep service and job logs clean.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
