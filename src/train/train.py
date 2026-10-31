"""Train the bike-demand Random Forest and save it to models/model.joblib.

Run from the repo root:
    python -m src.train.train
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, build_features

SEED = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # casual + registered = count, so they would leak the answer.
    df = df.drop(columns=["casual", "registered"], errors="ignore")
    X = build_features(df)
    y = np.log1p(df["count"])  # train on log scale, like the notebook
    return X, y


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )


def rmsle(y_true_log: pd.Series, y_pred_log: np.ndarray) -> float:
    y_true = np.expm1(y_true_log)
    y_pred = np.maximum(0, np.expm1(y_pred_log))
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def train(data_path: Path, out_dir: Path) -> dict:
    X, y = load_data(data_path)

    # Hold-out check so we know the model is sane before saving it.
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = make_model().fit(X_tr, y_tr)
    val_rmsle = rmsle(y_val, model.predict(X_val))

    # Final model on all the data.
    model = make_model().fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib", compress=3)
    metrics = {
        "model": "RandomForestRegressor",
        "val_rmsle": round(val_rmsle, 4),
        "n_rows": int(len(X)),
        "features": FEATURE_COLUMNS,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bike demand model")
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    metrics = train(args.data, args.out)
    print(f"Saved {args.out / 'model.joblib'}  validation RMSLE={metrics['val_rmsle']}")


if __name__ == "__main__":
    main()
