"""Pure prediction functions.

These take an already-loaded model plus the request inputs and return a numeric
prediction. Keeping them free of FastAPI / MLflow makes them trivially unit
testable with any sklearn-style estimator.
"""

from __future__ import annotations

import pandas as pd

from src.config import Config, get_config
from src.features import fare_features
from src.serving import features


def _fare_row(
    *,
    trip_distance: float,
    passenger_count: int,
    pu_location_id: int,
    do_location_id: int,
    pickup_datetime: pd.Timestamp,
    trip_duration_min: float | None,
    config: Config,
    duration_lookup=None,
    booking: bool = False,
) -> pd.DataFrame:
    row = features.build_fare_row(
        trip_distance=trip_distance,
        passenger_count=passenger_count,
        pu_location_id=pu_location_id,
        do_location_id=do_location_id,
        pickup_datetime=pickup_datetime,
        trip_duration_min=trip_duration_min,
        config=config,
        duration_lookup=duration_lookup,
    )
    # The booking-time model is trained without the trip duration.
    return row.drop(columns=list(fare_features.BOOKING_EXCLUDED)) if booking else row


def predict_fare(
    model,
    *,
    trip_distance: float,
    passenger_count: int,
    pu_location_id: int,
    do_location_id: int,
    pickup_datetime: pd.Timestamp,
    trip_duration_min: float | None = None,
    config: Config | None = None,
    duration_lookup=None,
    booking: bool = False,
) -> float:
    """Predict the fare amount for a single trip.

    ``booking=True`` builds the row for the booking-time model (no trip duration).
    """
    config = config or get_config()
    X = _fare_row(
        trip_distance=trip_distance,
        passenger_count=passenger_count,
        pu_location_id=pu_location_id,
        do_location_id=do_location_id,
        pickup_datetime=pickup_datetime,
        trip_duration_min=trip_duration_min,
        config=config,
        duration_lookup=duration_lookup,
        booking=booking,
    )
    return float(model.predict(X)[0])


def explain_fare(
    model,
    *,
    trip_distance: float,
    passenger_count: int,
    pu_location_id: int,
    do_location_id: int,
    pickup_datetime: pd.Timestamp,
    trip_duration_min: float | None = None,
    config: Config | None = None,
    explainer=None,
    duration_lookup=None,
    booking: bool = False,
):
    """SHAP explanation for a single fare prediction (same row the model sees)."""
    from src.responsible.explain import explain_instance

    config = config or get_config()
    X = _fare_row(
        trip_distance=trip_distance,
        passenger_count=passenger_count,
        pu_location_id=pu_location_id,
        do_location_id=do_location_id,
        pickup_datetime=pickup_datetime,
        trip_duration_min=trip_duration_min,
        config=config,
        duration_lookup=duration_lookup,
        booking=booking,
    )
    return explain_instance(model, X, explainer=explainer)


def predict_demand(
    model,
    history: pd.DataFrame,
    *,
    zone: int,
    target_hour: pd.Timestamp,
    config: Config | None = None,
) -> float:
    """Predict the pickup count for a single ``(zone, hour)``.

    Predictions are clipped at zero (pickup counts cannot be negative).
    """
    config = config or get_config()
    X = features.build_demand_row(history, zone, target_hour, config)
    pred = float(model.predict(X)[0])
    return max(0.0, pred)


def explain_demand(
    model,
    history: pd.DataFrame,
    *,
    zone: int,
    target_hour: pd.Timestamp,
    config: Config | None = None,
    explainer=None,
):
    """SHAP explanation for a single demand prediction (same row the model sees)."""
    from src.responsible.explain import explain_instance

    config = config or get_config()
    X = features.build_demand_row(history, zone, target_hour, config)
    return explain_instance(model, X, explainer=explainer)
