"""Model monitoring (drift + performance).

The serving layer runs on ECS Fargate and emits structured *prediction-capture*
log records; these are shipped to S3 and analysed on a schedule by a SageMaker
Processing job. This package provides the pure, dependency-light building blocks:

- ``drift``       - feature-distribution drift (KS test + PSI).
- ``performance`` - prediction accuracy vs. delayed actuals + degradation checks.
- ``capture``     - parse captured inference records into a DataFrame.

Heavy, AWS-only tooling (Evidently report rendering, boto3/CloudWatch, the
SageMaker SDK) is imported lazily / lives in the job entrypoints, so this package
stays importable and unit-testable with only numpy/scipy/pandas.
"""

from __future__ import annotations

from src.monitoring.drift import (
    DriftReport,
    FeatureDrift,
    compute_drift,
    population_stability_index,
)
from src.monitoring.performance import evaluate_performance, performance_degradation

__all__ = [
    "DriftReport",
    "FeatureDrift",
    "compute_drift",
    "population_stability_index",
    "evaluate_performance",
    "performance_degradation",
]
