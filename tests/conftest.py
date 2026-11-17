"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tests.synthetic import make_synthetic_trips


@pytest.fixture(scope="session", autouse=True)
def isolated_mlflow(tmp_path_factory):
    """Send every MLflow call in the test run to a throwaway store.

    MLflow 3 defaults to ``sqlite:///mlflow.db`` in the working directory, so a
    test that logs without configuring MLflow would otherwise write into the
    project's real registry.
    """
    import os

    import mlflow

    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    yield uri
    if previous is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = previous


@pytest.fixture(scope="session")
def raw_trips():
    """A synthetic raw trips DataFrame (includes dirty rows)."""
    return make_synthetic_trips()


@pytest.fixture()
def clean_trips_df(raw_trips):
    """Cleaned trips, produced via the real cleaning routine."""
    from src.data.validation import clean_trips

    clean, _ = clean_trips(raw_trips)
    return clean


@pytest.fixture()
def client(clean_trips_df):
    """FastAPI TestClient backed by small in-process models.

    Yields ``(test_client, demand_df)`` and registers the service for the
    duration of the test.
    """
    from fastapi.testclient import TestClient
    from sklearn.tree import DecisionTreeRegressor

    from src.data.aggregation import aggregate_hourly_demand
    from src.features.demand_features import build_demand_features
    from src.features.fare_features import build_fare_features
    from src.models import common
    from src.serving import api as api_module
    from src.serving import service as service_module
    from src.serving.service import ModelService

    demand = aggregate_hourly_demand(clean_trips_df)
    demand_feats = build_demand_features(demand)
    Xd, yd = common.split_to_xy(demand_feats, "demand")
    demand_model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xd, yd)

    fare_feats = build_fare_features(clean_trips_df)
    Xf, yf = common.split_to_xy(fare_feats, "fare")
    fare_model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xf, yf)

    svc = ModelService(demand_model, fare_model, demand)
    service_module.set_service(svc)
    with TestClient(api_module.app) as c:
        yield c, demand
    service_module.set_service(None)
