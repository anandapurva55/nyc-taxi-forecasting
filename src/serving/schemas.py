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
                "pickup_datetime": "2024-02-27T18:30:00",
            }
        }
    }


class FareResponse(BaseModel):
    predicted_fare: float = Field(..., description="Predicted fare amount (USD).")
    model: str = Field(
        ...,
        description="Registered model that answered: the fare model when trip_duration_min "
        "is sent, the booking-time model (trained without duration) when it is not.",
    )
    duration_estimated: bool = Field(
        ...,
        description="True if trip_duration_min was not supplied and had to be estimated "
        "(only when no booking-time model is registered).",
    )
    low_confidence: bool = Field(
        False,
        description="True when the trip has an unknown pickup or drop-off zone, the "
        "group with the largest errors in the subgroup audit.",
    )
    notes: list[str] = Field(default_factory=list, description="Reasons behind low_confidence.")
    request_id: str


class DemandRequest(BaseModel):
    """Inputs for a single demand (pickup-count) prediction."""

    zone: int = Field(..., ge=1, description="Pickup TLC zone ID.")
    target_hour: datetime = Field(
        ..., description="Hour to forecast (truncated to the hour, ISO 8601)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {"zone": 132, "target_hour": "2024-02-29T20:00:00"}
        }
    }


class DemandResponse(BaseModel):
    zone: int
    target_hour: datetime
    predicted_pickups: float = Field(..., description="Predicted pickup count (>= 0).")
    method: str = Field(
        ..., description='"model", or "same_hour_last_week" beyond the model horizon.'
    )
    request_id: str


class ForecastRequest(BaseModel):
    """Inputs for a multi-hour demand forecast."""

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
    method: str = Field(
        ..., description='"model", or "same_hour_last_week" beyond the model horizon.'
    )


class ForecastResponse(BaseModel):
    """A multi-hour demand forecast for one or more zones."""

    horizon_hours: int
    model_horizon_hours: int = Field(
        ..., description="Hours past the latest observed hour served by the model."
    )
    zones: int
    points: list[ForecastPoint]
    model_reference: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    demand_model_loaded: bool
    fare_model_loaded: bool
    fare_booking_model_loaded: bool = False
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


class VersionResponse(BaseModel):
    """Provenance of the running service."""

    git_commit: str = Field(..., description="Commit the image was built from.")
    built_at: str | None = None
    models: dict = Field(..., description="Registered model name and version per problem.")
    inputs: dict = Field(default_factory=dict, description="SHA-256 of baked input files.")
    fare_without_duration: str = Field(
        ..., description="How fares are quoted when trip_duration_min is not sent."
    )
    demand_model_horizon_hours: int = Field(
        ..., description="Hours ahead served by the demand model before the weekly rule."
    )
