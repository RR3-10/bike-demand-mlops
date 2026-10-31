"""FastAPI service for bike-demand predictions.

Run locally:
    uvicorn src.app.main:app --port 8000
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.features import build_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
# Set at image build time by CI (the git commit SHA), so we can prove
# which version is running. This is the fix for Lab 1, Twist 3.
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

state: dict = {}


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    yield
    state.clear()


app = FastAPI(title="Bike Sharing Demand API", version="1.0.0", lifespan=lifespan)


class RideConditions(BaseModel):
    """One hour of conditions. Field names match the Kaggle CSV."""

    model_config = ConfigDict(
        extra="forbid",  # a renamed/unknown field is rejected, not silently ignored
        json_schema_extra={
            "example": {
                "datetime": "2012-06-15 08:00:00",
                "season": 2,
                "holiday": 0,
                "workingday": 1,
                "weather": 1,
                "temp": 24.6,
                "humidity": 60,
                "windspeed": 12.998,
            }
        },
    )

    datetime: datetime
    season: Literal[1, 2, 3, 4] = Field(description="1 spring, 2 summer, 3 fall, 4 winter")
    holiday: Literal[0, 1]
    workingday: Literal[0, 1]
    weather: Literal[1, 2, 3, 4] = Field(description="1 clear ... 4 heavy rain/snow")
    temp: float = Field(ge=-20, le=50, description="Temperature in Celsius")
    humidity: float = Field(ge=0, le=100)
    windspeed: float = Field(ge=0, le=100)


class Prediction(BaseModel):
    predicted_count: int
    model_version: str


@app.get("/health")
def health() -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(req: RideConditions) -> Prediction:
    X = build_features(pd.DataFrame([req.model_dump()]))
    y_log = state["model"].predict(X)[0]
    count = max(0, int(round(float(np.expm1(y_log)))))
    return Prediction(predicted_count=count, model_version=MODEL_VERSION)
