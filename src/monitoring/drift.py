"""Feature-distribution drift detection (KS test + Population Stability Index).

Given a *reference* sample (typically the training feature distribution) and a
*current* sample (recent captured inference inputs), we quantify drift per
feature using two complementary signals:

- **PSI** (Population Stability Index): bins the reference distribution and
  measures how much probability mass has shifted. Rules of thumb:
  ``< 0.1`` no shift, ``0.1-0.2`` moderate, ``>= 0.2`` significant drift.
- **KS two-sample test**: distribution-free test; a small p-value indicates the
  two samples are unlikely to share a distribution.

The ``drifted`` flag is driven primarily by PSI (stable across sample sizes);
the KS statistic/p-value are reported alongside for context.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_KS_ALPHA = 0.05
DEFAULT_BINS = 10


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = DEFAULT_BINS
) -> float:
    """Compute the PSI of ``current`` relative to ``reference``.

    Bin edges are the quantiles of the reference sample. A small epsilon guards
    empty bins so the log ratio stays finite. Returns 0.0 when the reference has
    no spread (a constant feature).
    """
    reference = np.asarray(reference, dtype="float64")
    reference = reference[~np.isnan(reference)]
    current = np.asarray(current, dtype="float64")
    current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # Constant (or near-constant) reference: no meaningful bins.
        return 0.0
    # Open the outer edges so out-of-range current values are still counted.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class FeatureDrift:
    """Per-feature drift result."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    """Aggregate drift across all evaluated features."""

    features: list[FeatureDrift] = field(default_factory=list)
    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    ks_alpha: float = DEFAULT_KS_ALPHA

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_drifted(self) -> int:
        return sum(f.drifted for f in self.features)

    @property
    def share_drifted(self) -> float:
        return self.n_drifted / self.n_features if self.features else 0.0

    @property
    def drifted(self) -> bool:
        """Overall drift = at least one feature drifted."""
        return self.n_drifted > 0

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_drifted": self.n_drifted,
            "share_drifted": self.share_drifted,
            "drifted": self.drifted,
            "psi_threshold": self.psi_threshold,
            "ks_alpha": self.ks_alpha,
            "features": [f.to_dict() for f in self.features],
        }

    def cloudwatch_metrics(self, namespace: str = "TaxiForecasting/Drift") -> list[dict]:
        """Shape the summary as CloudWatch ``put_metric_data`` entries."""
        return [
            {"MetricName": "FeaturesDrifted", "Value": float(self.n_drifted), "Unit": "Count"},
            {"MetricName": "ShareDrifted", "Value": float(self.share_drifted), "Unit": "None"},
            {
                "MetricName": "MaxPSI",
                "Value": max((f.psi for f in self.features), default=0.0),
                "Unit": "None",
            },
        ]


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str] | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
    bins: int = DEFAULT_BINS,
) -> DriftReport:
    """Compare ``current`` against ``reference`` over the given numeric features.

    If ``features`` is omitted, the numeric columns common to both frames are
    used. Raises ``ValueError`` when no comparable features are found.
    """
    if features is None:
        common = [c for c in reference.columns if c in current.columns]
        features = [c for c in common if pd.api.types.is_numeric_dtype(reference[c])]
    features = list(features)
    if not features:
        raise ValueError("No numeric features in common between reference and current.")

    results: list[FeatureDrift] = []
    for col in features:
        ref = pd.to_numeric(reference[col], errors="coerce").to_numpy()
        cur = pd.to_numeric(current[col], errors="coerce").to_numpy()
        psi = population_stability_index(ref, cur, bins=bins)
        ref_clean = ref[~np.isnan(ref)]
        cur_clean = cur[~np.isnan(cur)]
        if ref_clean.size and cur_clean.size:
            ks = ks_2samp(ref_clean, cur_clean)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        else:
            ks_stat, ks_p = 0.0, 1.0
        results.append(
            FeatureDrift(
                feature=col,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                drifted=bool(psi >= psi_threshold),
            )
        )

    return DriftReport(features=results, psi_threshold=psi_threshold, ks_alpha=ks_alpha)


def build_evidently_report(reference: pd.DataFrame, current: pd.DataFrame):  # pragma: no cover
    """Build an Evidently data-drift report (HTML-renderable).

    Evidently is an AWS-job-only dependency (see ``requirements-monitoring.txt``);
    it is imported lazily so this module stays lightweight for serving/tests.
    """
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "evidently is not installed; add requirements-monitoring.txt in the job image."
        ) from exc

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report
