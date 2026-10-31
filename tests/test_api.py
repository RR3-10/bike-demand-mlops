"""Tests for the bike-demand service. Each test says what it protects."""

import pandas as pd

from src.app.main import load_model
from src.features import FEATURE_COLUMNS, build_features


def test_health_endpoint(client):
    """API contract: /health answers 200 so Docker and deploys know we are alive."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "model_version" in r.json()


def test_predict_contract(client, valid_request):
    """API contract: /predict returns a non-negative whole number of bikes and a version."""
    r = client.post("/predict", json=valid_request)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predicted_count", "model_version"}
    assert isinstance(body["predicted_count"], int)
    assert body["predicted_count"] >= 0


def test_model_loads_and_matches_features(trained_model):
    """Model loads: the saved file opens and expects exactly our feature columns."""
    model = load_model(trained_model)
    assert hasattr(model, "predict")
    assert model.n_features_in_ == len(FEATURE_COLUMNS)


def test_renamed_field_is_rejected(client, valid_request):
    """Input schema: a renamed column (Lab 1, Twist 1) gives a clear 422, not a crash."""
    bad = dict(valid_request)
    bad["temperature"] = bad.pop("temp")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_out_of_range_values_are_rejected(client, valid_request):
    """Input schema: impossible values (season 5, humidity 150) are refused."""
    assert client.post("/predict", json={**valid_request, "season": 5}).status_code == 422
    assert client.post("/predict", json={**valid_request, "humidity": 150}).status_code == 422


def test_edge_case_extreme_weather_at_midnight(client, valid_request):
    """Edge case: rare category (weather 4) at hour 0 with boundary values still works."""
    edge = {
        **valid_request,
        "datetime": "2011-01-01 00:00:00",
        "season": 4,
        "weather": 4,
        "temp": -20,
        "humidity": 100,
        "windspeed": 0,
    }
    r = client.post("/predict", json=edge)
    assert r.status_code == 200
    assert r.json()["predicted_count"] >= 0


def test_single_row_features_match_columns(valid_request):
    """Feature engineering: one request produces every column, in training order."""
    X = build_features(pd.DataFrame([valid_request]))
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.loc[0, "is_rush_hour"] == 1  # 08:00 is a rush hour
    assert X.loc[0, "season_2"] == 1


def test_prediction_is_deterministic(client, valid_request):
    """Deterministic prediction: same input gives the same answer, and rush hour beats 3 AM."""
    first = client.post("/predict", json=valid_request).json()["predicted_count"]
    second = client.post("/predict", json=valid_request).json()["predicted_count"]
    assert first == second

    night = {**valid_request, "datetime": "2012-06-15 03:00:00"}
    night_count = client.post("/predict", json=night).json()["predicted_count"]
    assert first > night_count
