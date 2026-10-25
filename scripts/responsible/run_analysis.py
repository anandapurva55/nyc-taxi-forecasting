"""Generate the responsible-AI artifacts.

For each problem (demand, fare) this:

1. loads the registered model and the held-out **test** split;
2. computes overall regression metrics;
3. slices performance by **borough**, **time-of-day**, **weekend/holiday** and
   **airport** zones, flagging any subgroup disparity;
4. computes SHAP global feature importance (+ a summary plot); and
5. writes a Markdown **model card** (docs/model_cards/) plus a fairness JSON and
   SHAP plot (artifacts/responsible/, gitignored).

Run locally after training::

    python -m scripts.responsible.run_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import Config, get_config
from src.data.ingestion import load_zone_lookup
from src.models.common import PROBLEMS, registry_model_name, split_to_xy
from src.responsible import fairness
from src.responsible.explain import global_importance, save_summary_plot
from src.responsible.model_card import render_model_card
from src.serving import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("responsible-analysis")

SHAP_SAMPLE = 2000
CARDS_DIR = Path("docs/model_cards")
ARTIFACTS_DIR = Path("artifacts/responsible")


def _load_test_frame(config: Config, problem: str) -> pd.DataFrame:
    prefix = PROBLEMS[problem]["file_prefix"]
    path = config.features_dir / f"{prefix}_test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing test split: {path}. Run the pipeline first.")
    return pd.read_parquet(path)


def _build_slices(df: pd.DataFrame, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach the subgroup label columns used for fairness slicing."""
    slices = fairness.add_borough(df, zone_lookup, location_col="PULocationID")
    slices["time_of_day"] = slices["hour"].apply(fairness.time_of_day_bucket)
    slices["day_type"] = slices["is_weekend"].map({1: "weekend", 0: "weekday"})
    slices["holiday"] = slices["is_holiday"].map({1: "holiday", 0: "non-holiday"})
    slices["airport"] = (slices["service_zone"] == "Airports").map(
        {True: "airport", False: "non-airport"}
    )
    return slices


def analyse_problem(problem: str, config: Config, zone_lookup: pd.DataFrame) -> dict:
    logger.info("Analysing %s", problem)
    model = registry.load_model(problem, config)
    df = _load_test_frame(config, problem)
    X, y = split_to_xy(df, problem)
    y_pred = model.predict(X)

    from src.models.common import regression_metrics

    metrics = regression_metrics(y.to_numpy(), y_pred)

    slices = _build_slices(df, zone_lookup)
    dimensions = ["borough", "time_of_day", "day_type", "holiday", "airport"]
    reports = [
        fairness.subgroup_metrics(slices, y.to_numpy(), y_pred, dim) for dim in dimensions
    ]

    # SHAP on a bounded sample for speed.
    sample = X.sample(min(SHAP_SAMPLE, len(X)), random_state=42)
    importance = global_importance(model, sample)
    top_features = list(importance.head(10).itertuples(index=False, name=None))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        save_summary_plot(model, sample, ARTIFACTS_DIR / f"{problem}_shap_summary.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("SHAP summary plot skipped for %s: %s", problem, exc)

    fairness_json = {
        "problem": problem,
        "metrics": metrics,
        "dimensions": [r.to_dict() for r in reports],
    }
    (ARTIFACTS_DIR / f"{problem}_fairness.json").write_text(json.dumps(fairness_json, indent=2))

    card = render_model_card(
        name=registry_model_name(problem, config),
        problem=problem,
        algorithm=type(model).__name__,
        target=PROBLEMS[problem]["target"],
        metrics=metrics,
        fairness=reports,
        top_features=top_features,
    )
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    (CARDS_DIR / f"{problem}.md").write_text(card, encoding="utf-8")
    logger.info(
        "%s: wrote model card + fairness JSON (%d flagged dimensions)",
        problem,
        sum(r.flagged for r in reports),
    )
    return fairness_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate responsible-AI artifacts.")
    parser.add_argument("--problem", choices=list(PROBLEMS), default=None, help="Limit to one problem.")
    args = parser.parse_args(argv)

    config = get_config()
    zone_lookup = load_zone_lookup(config.raw_dir / "taxi_zone_lookup.csv")

    problems = [args.problem] if args.problem else list(PROBLEMS)
    for problem in problems:
        analyse_problem(problem, config, zone_lookup)
    logger.info("Responsible-AI analysis complete. Cards in %s", CARDS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
