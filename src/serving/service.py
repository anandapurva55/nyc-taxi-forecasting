"""ModelService - holds the loaded models and demand history for serving.

Encapsulates the runtime state the API needs (the registered models plus the
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
        *,
        duration_lookup=None,
        model_versions: dict | None = None,
        fare_booking_model=None,
    ) -> None:
        self.config = config or get_config()
        self.demand_model = demand_model
        self.fare_model = fare_model
        self.demand_history = demand_history
        # Fare model without trip duration, used when the caller does not send one.
        self.fare_booking_model = fare_booking_model
        # Learned duration estimate, used only if no booking-time model is registered.
        self.duration_lookup = duration_lookup
        # Exact registered versions being served, e.g. {"fare": {"version": 2, ...}}.
        self.model_versions = model_versions or {}
        # SHAP TreeExplainers are built lazily on first /explain call and cached.
        self._fare_explainer = None
        self._fare_booking_explainer = None
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
        try:
            booking_model = registry.load_model("fare_booking", config)
        except Exception as exc:  # registries trained before the booking model existed
            logger.warning("No booking-time fare model; estimating trip duration instead: %s", exc)
            booking_model = None
        history = load_demand_history(config)
        problems = ("demand", "fare", "fare_booking") if booking_model is not None else ("demand", "fare")
        versions = {
            problem: info
            for problem in problems
            if (info := registry.resolve_version(problem, config)) is not None
        }
        lookup = None if booking_model is not None else registry.load_duration_lookup(config)
        service = cls(
            demand_model, fare_model, history, config,
            duration_lookup=lookup, model_versions=versions, fare_booking_model=booking_model,
        )
        logger.info(
            "ModelService ready: %d demand-history rows, %d zones, versions %s, fare without duration: %s",
            len(history),
            history["PULocationID"].nunique(),
            {p: v["version"] for p, v in versions.items()},
            service.fare_without_duration(),
        )
        return service

    def model_version(self, problem: str) -> str:
        """Exact served version label, e.g. "taxi-fare-predictor/2"."""
        info = self.model_versions.get(problem)
        if info is None:
            return str(self.config.serving.get("model_stage", "latest"))
        return f"{info['name']}/{info['version']}"

    def fare_without_duration(self) -> str:
        """How fares are quoted when the caller does not send the trip duration."""
        if self.fare_booking_model is not None:
            return "booking-time model"
        return "learned duration estimate" if self.duration_lookup is not None else "fixed duration estimate"

    def fare_problem(self, trip_duration_min: float | None) -> str:
        """Which registered fare model answers a request."""
        if trip_duration_min is None and self.fare_booking_model is not None:
            return "fare_booking"
        return "fare"

    # --- Predictions -------------------------------------------------------
    def predict_fare(self, **kwargs) -> float:
        if self.fare_problem(kwargs.get("trip_duration_min")) == "fare_booking":
            return predict.predict_fare(self.fare_booking_model, config=self.config, booking=True, **kwargs)
        return predict.predict_fare(
            self.fare_model, config=self.config, duration_lookup=self.duration_lookup, **kwargs
        )

    def demand_method(self, target_hour: pd.Timestamp) -> str:
        """Whether the model or the same-hour-last-week rule serves this hour."""
        from src.serving import batch

        ahead = batch.hours_ahead(pd.Timestamp(target_hour).floor("h"), self.latest_history_hour())
        return batch.SEASONAL_NAIVE if ahead > batch.model_horizon(self.config) else batch.MODEL

    def _routed_demand(self, zone: int, target_hour: pd.Timestamp) -> float:
        from src.serving import batch

        last = self.latest_history_hour()
        ahead = batch.hours_ahead(target_hour, last)
        if ahead <= 0:  # inside the history: every lag is observed
            return predict.predict_demand(
                self.demand_model, self.demand_history, zone=zone, target_hour=target_hour, config=self.config
            )
        forecast = batch.forecast_demand(self, horizon_hours=ahead, zones=[zone])
        return float(forecast["predicted_pickups"].iloc[-1])

    def predict_demand(self, *, zone: int, target_hour: pd.Timestamp) -> float:
        normalized_hour = pd.Timestamp(target_hour).floor("h")
        cache_key = (int(zone), normalized_hour)
        with self._demand_cache_lock:
            cached = self._demand_prediction_cache.get(cache_key)
            if cached is not None:
                self._demand_prediction_cache.move_to_end(cache_key)
                return cached

        predicted = self._routed_demand(cache_key[0], normalized_hour)
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

    def fare_booking_explainer(self):
        if self._fare_booking_explainer is None:
            from src.responsible.explain import build_explainer

            self._fare_booking_explainer = build_explainer(self.fare_booking_model)
        return self._fare_booking_explainer

    def demand_explainer(self):
        if self._demand_explainer is None:
            from src.responsible.explain import build_explainer

            self._demand_explainer = build_explainer(self.demand_model)
        return self._demand_explainer

    def explain_fare(self, **kwargs):
        if self.fare_problem(kwargs.get("trip_duration_min")) == "fare_booking":
            return predict.explain_fare(
                self.fare_booking_model,
                config=self.config,
                explainer=self.fare_booking_explainer(),
                booking=True,
                **kwargs,
            )
        return predict.explain_fare(
            self.fare_model,
            config=self.config,
            explainer=self.fare_explainer(),
            duration_lookup=self.duration_lookup,
            **kwargs,
        )

    def explain_demand(self, *, zone: int, target_hour: pd.Timestamp):
        from src.serving import batch

        last = self.latest_history_hour()
        if batch.hours_ahead(pd.Timestamp(target_hour).floor("h"), last) > 1:
            raise ValueError(
                f"Explanations cover hours up to {last + pd.Timedelta(hours=1)} (one hour past the "
                "latest observed hour), where every model input is observed."
            )
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
