"""ModelService - holds the loaded models and demand history for serving.

Encapsulates the runtime state the API needs (the two registered models plus the
dense demand history used to build lag features) behind a small interface. The
FastAPI app depends on this via a provider function so tests can inject a fake
service and run entirely offline.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import pandas as pd

from src.config import Config, get_config
from src.serving import predict, registry

logger = logging.getLogger(__name__)


class ModelService:
    """Loaded models + demand history, with thin predict wrappers."""

    def __init__(
        self,
        demand_model,
        fare_model,
        demand_history: pd.DataFrame,
        config: Config | None = None,
    ) -> None:
        self.config = config or get_config()
        self.demand_model = demand_model
        self.fare_model = fare_model
        self.demand_history = demand_history
        # SHAP TreeExplainers are built lazily on first /explain call and cached.
        self._fare_explainer = None
        self._demand_explainer = None
        self._demand_prediction_cache: OrderedDict[tuple[int, pd.Timestamp], float] = OrderedDict()
        self._demand_cache_lock = Lock()

    # --- Constructors ------------------------------------------------------
    @classmethod
    def from_registry(cls, config: Config | None = None) -> ModelService:
        """Load both models from the MLflow registry and the demand history."""
        config = config or get_config()
        demand_model = registry.load_model("demand", config)
        fare_model = registry.load_model("fare", config)
        history = load_demand_history(config)
        logger.info(
            "ModelService ready: %d demand-history rows, %d zones",
            len(history),
            history["PULocationID"].nunique(),
        )
        return cls(demand_model, fare_model, history, config)

    # --- Predictions -------------------------------------------------------
    def predict_fare(self, **kwargs) -> float:
        return predict.predict_fare(self.fare_model, config=self.config, **kwargs)

    def predict_demand(self, *, zone: int, target_hour: pd.Timestamp) -> float:
        normalized_hour = pd.Timestamp(target_hour).floor("h")
        cache_key = (int(zone), normalized_hour)
        with self._demand_cache_lock:
            cached = self._demand_prediction_cache.get(cache_key)
            if cached is not None:
                self._demand_prediction_cache.move_to_end(cache_key)
                return cached

        predicted = predict.predict_demand(
            self.demand_model,
            self.demand_history,
            zone=cache_key[0],
            target_hour=normalized_hour,
            config=self.config,
        )
        with self._demand_cache_lock:
            self._demand_prediction_cache[cache_key] = predicted
            self._demand_prediction_cache.move_to_end(cache_key)
            if len(self._demand_prediction_cache) > 4096:
                self._demand_prediction_cache.popitem(last=False)
        return predicted

    def known_zones(self) -> list[int]:
        return sorted(int(z) for z in self.demand_history["PULocationID"].unique())

    # --- Explanations ------------------------------------------------------
    def fare_explainer(self):
        if self._fare_explainer is None:
            from src.responsible.explain import build_explainer

            self._fare_explainer = build_explainer(self.fare_model)
        return self._fare_explainer

    def demand_explainer(self):
        if self._demand_explainer is None:
            from src.responsible.explain import build_explainer

            self._demand_explainer = build_explainer(self.demand_model)
        return self._demand_explainer

    def explain_fare(self, **kwargs):
        return predict.explain_fare(
            self.fare_model, config=self.config, explainer=self.fare_explainer(), **kwargs
        )

    def explain_demand(self, *, zone: int, target_hour: pd.Timestamp):
        return predict.explain_demand(
            self.demand_model,
            self.demand_history,
            zone=zone,
            target_hour=target_hour,
            config=self.config,
            explainer=self.demand_explainer(),
        )

    def latest_history_hour(self) -> pd.Timestamp:
        return pd.Timestamp(self.demand_history["pickup_hour"].max())


def load_demand_history(config: Config | None = None) -> pd.DataFrame:
    """Load the dense hourly demand series used for lag features."""
    config = config or get_config()
    path = Path(config.serving.get("demand_history_path", "data/processed/demand_hourly.parquet"))
    if not path.is_absolute():
        from src.config import PROJECT_ROOT

        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Demand history not found: {path}. Run the pipeline first."
        )
    df = pd.read_parquet(path)
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    return df


# --- FastAPI dependency provider ------------------------------------------
_service: ModelService | None = None


def get_service() -> ModelService:
    """Return the process-wide ModelService, loading it on first use."""
    global _service
    if _service is None:
        _service = ModelService.from_registry()
    return _service


def set_service(service: ModelService | None) -> None:
    """Override (or reset) the cached service - used for tests and startup."""
    global _service
    _service = service
