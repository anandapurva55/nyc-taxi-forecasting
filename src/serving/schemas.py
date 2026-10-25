"""Pydantic v2 request/response schemas for the prediction API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FareRequest(BaseModel):
    """Inputs for a single fare prediction."""

    trip_distance: float = Field(..., gt=0, le=100, description="Trip distance in miles.")
    pu_location_id: int = Field(..., ge=1, description="Pickup TLC zone ID.")
    do_location_id: int = Field(..., ge=1, description="Drop-off TLC zone ID.")
    passenger_count: int = Field(1, ge=1, le=8, description="Number of passengers.")
    pickup_datetime: datetime = Field(..., description="Pickup timestamp (ISO 8601).")
    trip_duration_min: float | None = Field(
        None,
        gt=0,
        le=240,
        description="Optional trip duration (min). Estimated from distance if omitted.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "trip_distance": 3.4,
                "pu_location_id": 132,
                "do_location_id": 230,
                "passenger_count": 1,
                "pickup_datetime": "2024-01-15T18:30:00",
            }
        }
    }


class FareResponse(BaseModel):
    predicted_fare: float = Field(..., description="Predicted fare amount (USD).")
    duration_estimated: bool = Field(
        ..., description="True if trip_duration_min was estimated from distance."
    )
    request_id: str


class DemandRequest(BaseModel):
    """Inputs for a single demand (pickup-count) prediction."""

    zone: int = Field(..., ge=1, description="Pickup TLC zone ID.")
    target_hour: datetime = Field(
        ..., description="Hour to forecast (truncated to the hour, ISO 8601)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {"zone": 132, "target_hour": "2024-01-31T20:00:00"}
        }
    }


class DemandResponse(BaseModel):
    zone: int
    target_hour: datetime
    predicted_pickups: float = Field(..., description="Predicted pickup count (>= 0).")
    request_id: str


class ForecastRequest(BaseModel):
    """Inputs for a multi-hour recursive demand forecast."""

    horizon_hours: int = Field(
        24, ge=1, le=168, description="Number of hours to forecast ahead."
    )
    zones: list[int] | None = Field(
        None, description="Zones to forecast; defaults to all known zones."
    )
    start_hour: datetime | None = Field(
        None,
        description="First hour to forecast (ISO 8601); defaults to the hour after "
        "the latest history hour.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"horizon_hours": 6, "zones": [132, 230]}
        }
    }


class ForecastPoint(BaseModel):
    zone: int
    pickup_hour: datetime
    predicted_pickups: float = Field(..., description="Predicted pickup count (>= 0).")


class ForecastResponse(BaseModel):
    """A recursive multi-hour demand forecast for one or more zones."""

    horizon_hours: int
    zones: int
    points: list[ForecastPoint]
    model_reference: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    demand_model_loaded: bool
    fare_model_loaded: bool
    history_hours: int
    zones: int


class FeatureContribution(BaseModel):
    feature: str
    value: float = Field(..., description="Signed SHAP contribution for this feature.")


class ExplainResponse(BaseModel):
    """SHAP explanation of a single prediction (additive: base + sum = prediction)."""

    problem: str
    prediction: float = Field(..., description="Model output being explained.")
    base_value: float = Field(..., description="Explainer expected value (baseline).")
    contributions: list[FeatureContribution] = Field(
        ..., description="Per-feature SHAP contributions, ranked by absolute impact."
    )
    request_id: str
