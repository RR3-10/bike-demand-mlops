from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.main import MODEL_PATH, app
from src.train.train import train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Train once if the model file is missing (e.g. on a fresh CI runner)."""
    if not MODEL_PATH.exists():
        train(ROOT / "data" / "train.csv", MODEL_PATH.parent)
    return MODEL_PATH


@pytest.fixture()
def client():
    with TestClient(app) as c:  # "with" runs the startup that loads the model
        yield c


@pytest.fixture()
def valid_request() -> dict:
    return {
        "datetime": "2012-06-15 08:00:00",
        "season": 2,
        "holiday": 0,
        "workingday": 1,
        "weather": 1,
        "temp": 24.6,
        "humidity": 60,
        "windspeed": 12.998,
    }
