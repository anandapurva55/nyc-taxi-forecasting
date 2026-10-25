"""FastAPI application exposing demand and fare predictions.

Endpoints:
- ``GET  /health``          - liveness + model/state readiness.
- ``POST /predict/fare``    - single-trip fare prediction.
- ``POST /predict/demand``  - single ``(zone, hour)`` pickup-count prediction.
- ``POST /forecast/demand`` - recursive multi-hour pickup forecast per zone.

Each request is tagged with an ``X-Request-ID`` (generated if absent) for tracing,
and domain errors (e.g. unknown zone, insufficient history) are returned as 400s.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving import batch as batch_module
from src.serving import service as service_module
from src.serving.schemas import (
    DemandRequest,
    DemandResponse,
    ExplainResponse,
    FareRequest,
    FareResponse,
    FeatureContribution,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from src.serving.service import ModelService, get_service

logger = logging.getLogger(__name__)

# Dedicated logger for prediction-capture records (drift/performance
# monitoring). Emits one JSON line per prediction with ``event="prediction"``;
# Kinesis Firehose ships these to S3 where the drift job consumes them.
capture_logger = logging.getLogger("taxi.capture")
capture_logger.setLevel(logging.INFO)
capture_logger.propagate = False
if not capture_logger.handlers:
    capture_handler = logging.StreamHandler()
    capture_handler.setFormatter(logging.Formatter("%(message)s"))
    capture_logger.addHandler(capture_handler)


def _model_stage(svc: ModelService) -> str:
    try:
        return str(svc.config.serving.get("model_stage", "latest"))
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _capture_prediction(
    request_id: str, problem: str, model_version: str, features: dict, prediction: float
) -> None:
    """Emit a structured prediction-capture record for downstream drift analysis."""
    capture_logger.info(
        json.dumps(
            {
                "event": "prediction",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "problem": problem,
                "model_version": model_version,
                "features": features,
                "prediction": prediction,
            }
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup unless a service was already injected (e.g. tests).
    if service_module._service is None:
        try:
            service_module.get_service()
        except Exception as exc:  # pragma: no cover - depends on registry availability
            logger.warning("Model service not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="NYC Taxi Demand & Fare Prediction API",
    version="1.0.0",
    description="Serves the demand-forecasting and fare-prediction models.",
    lifespan=lifespan,
)

# Prometheus /metrics endpoint (monitoring): request count,
# latency histograms, and in-progress gauges, exposed for scraping.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    # Structured JSON access log, one line per request, for log aggregation.
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )
    )
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "request_id": _request_id(request)},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = service_module._service
    if svc is None:
        return HealthResponse(
            status="degraded",
            demand_model_loaded=False,
            fare_model_loaded=False,
            history_hours=0,
            zones=0,
        )
    history_hours = int(svc.demand_history["pickup_hour"].nunique())
    return HealthResponse(
        status="ok",
        demand_model_loaded=svc.demand_model is not None,
        fare_model_loaded=svc.fare_model is not None,
        history_hours=history_hours,
        zones=len(svc.known_zones()),
    )


@app.post("/predict/fare", response_model=FareResponse)
def predict_fare(
    payload: FareRequest,
    request: Request,
    svc: ModelService = Depends(get_service),
) -> FareResponse:
    predicted = svc.predict_fare(
        trip_distance=payload.trip_distance,
        passenger_count=payload.passenger_count,
        pu_location_id=payload.pu_location_id,
        do_location_id=payload.do_location_id,
        pickup_datetime=payload.pickup_datetime,
        trip_duration_min=payload.trip_duration_min,
    )
    _capture_prediction(
        request_id=_request_id(request),
        problem="fare",
        model_version=_model_stage(svc),
        features={
            "trip_distance": payload.trip_distance,
            "passenger_count": payload.passenger_count,
            "pu_location_id": payload.pu_location_id,
            "do_location_id": payload.do_location_id,
            "trip_duration_min": payload.trip_duration_min,
        },
        prediction=round(predicted, 2),
    )
    return FareResponse(
        predicted_fare=round(predicted, 2),
        duration_estimated=payload.trip_duration_min is None,
        request_id=_request_id(request),
    )


@app.post("/predict/demand", response_model=DemandResponse)
def predict_demand(
    payload: DemandRequest,
    request: Request,
    svc: ModelService = Depends(get_service),
) -> DemandResponse:
    if payload.zone not in svc.known_zones():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown zone {payload.zone}; no demand history available.",
        )
    predicted = svc.predict_demand(
        zone=payload.zone, target_hour=pd.Timestamp(payload.target_hour)
    )
    target_ts = pd.Timestamp(payload.target_hour)
    _capture_prediction(
        request_id=_request_id(request),
        problem="demand",
        model_version=_model_stage(svc),
        features={
            "zone": payload.zone,
            "hour_of_day": int(target_ts.hour),
            "day_of_week": int(target_ts.dayofweek),
        },
        prediction=round(predicted, 2),
    )
    return DemandResponse(
        zone=payload.zone,
        target_hour=payload.target_hour,
        predicted_pickups=round(predicted, 2),
        request_id=_request_id(request),
    )


@app.post("/forecast/demand", response_model=ForecastResponse)
def forecast_demand(
    payload: ForecastRequest,
    request: Request,
    svc: ModelService = Depends(get_service),
) -> ForecastResponse:
    """Recursive multi-hour pickup forecast for one or more zones.

    Wraps the batch forecaster so the dashboard can render time-series and zone
    heatmaps without replicating the recursive lag logic client-side.
    """
    zones = payload.zones
    if zones is not None:
        unknown = [z for z in zones if z not in svc.known_zones()]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown zone(s) {unknown}; no demand history available.",
            )
    forecasts = batch_module.forecast_demand(
        svc,
        horizon_hours=payload.horizon_hours,
        start_hour=pd.Timestamp(payload.start_hour) if payload.start_hour else None,
        zones=zones,
    )
    points = [
        ForecastPoint(
            zone=int(row.PULocationID),
            pickup_hour=pd.Timestamp(row.pickup_hour).to_pydatetime(),
            predicted_pickups=round(float(row.predicted_pickups), 2),
        )
        for row in forecasts.itertuples(index=False)
    ]
    return ForecastResponse(
        horizon_hours=payload.horizon_hours,
        zones=int(forecasts["PULocationID"].nunique()) if not forecasts.empty else 0,
        points=points,
        model_reference=_model_stage(svc),
        request_id=_request_id(request),
    )


def _explain_response(problem: str, explanation, request: Request) -> ExplainResponse:
    contributions = [
        FeatureContribution(feature=name, value=value)
        for name, value in sorted(
            explanation.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
    ]
    return ExplainResponse(
        problem=problem,
        prediction=round(explanation.prediction, 4),
        base_value=round(explanation.base_value, 4),
        contributions=contributions,
        request_id=_request_id(request),
    )


@app.post("/explain/fare", response_model=ExplainResponse)
def explain_fare(
    payload: FareRequest,
    request: Request,
    svc: ModelService = Depends(get_service),
) -> ExplainResponse:
    explanation = svc.explain_fare(
        trip_distance=payload.trip_distance,
        passenger_count=payload.passenger_count,
        pu_location_id=payload.pu_location_id,
        do_location_id=payload.do_location_id,
        pickup_datetime=payload.pickup_datetime,
        trip_duration_min=payload.trip_duration_min,
    )
    return _explain_response("fare", explanation, request)


@app.post("/explain/demand", response_model=ExplainResponse)
def explain_demand(
    payload: DemandRequest,
    request: Request,
    svc: ModelService = Depends(get_service),
) -> ExplainResponse:
    if payload.zone not in svc.known_zones():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown zone {payload.zone}; no demand history available.",
        )
    explanation = svc.explain_demand(
        zone=payload.zone, target_hour=pd.Timestamp(payload.target_hour)
    )
    return _explain_response("demand", explanation, request)
