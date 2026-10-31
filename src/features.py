"""Feature engineering shared by training and serving.

Same steps as the original Bike_sharing_demand notebook, written so they work
on one row (an API request) as well as on the whole training set.
"""

import numpy as np
import pandas as pd

# Columns the raw data must have (atemp is optional: the notebook dropped it).
RAW_COLUMNS = [
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "humidity",
    "windspeed",
]

# Final feature order the model is trained on. Serving uses the same list.
FEATURE_COLUMNS = [
    "holiday",
    "workingday",
    "temp",
    "humidity",
    "windspeed",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "season_2",
    "season_3",
    "season_4",
    "weather_2",
    "weather_3",
    "weather_4",
    "temp_humidity",
    "hour_sin",
    "hour_cos",
    "working_hour",
]

RUSH_HOURS = [7, 8, 17, 18]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw rows into the model's feature matrix."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["datetime"])

    out["holiday"] = df["holiday"].astype(int)
    out["workingday"] = df["workingday"].astype(int)
    out["temp"] = df["temp"].astype(float)
    out["humidity"] = df["humidity"].astype(float)
    out["windspeed"] = df["windspeed"].astype(float)

    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.weekday
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_rush_hour"] = out["hour"].isin(RUSH_HOURS).astype(int)

    # One-hot with drop_first (season 1 and weather 1 are the baseline).
    # Written by hand so a single request still gets every column.
    for s in (2, 3, 4):
        out[f"season_{s}"] = (df["season"].astype(int) == s).astype(int)
    for w in (2, 3, 4):
        out[f"weather_{w}"] = (df["weather"].astype(int) == w).astype(int)

    out["temp_humidity"] = out["temp"] * out["humidity"]
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["working_hour"] = out["workingday"] * out["hour"]

    return out[FEATURE_COLUMNS]
