# Bike Sharing Demand API

[![CI](https://github.com/RR3-10/bike-demand-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/RR3-10/bike-demand-mlops/actions/workflows/ci.yml)

AI381 Lab 2 · Rashed Moukattash

Predicts how many bikes will be rented in one hour from the date, time and weather.
The model is my Random Forest from the Kaggle *Bike Sharing Demand* project (AI249/AI328),
moved out of the notebook into a tested FastAPI service that runs in Docker and is
built, scanned and published by GitHub Actions on every commit to `main`.

## Run it (one command)

```bash
docker build -t bike-demand . && docker run --rm -p 8000:8000 bike-demand
```

Then open <http://localhost:8000/docs>, or:

```bash
curl localhost:8000/health

curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"datetime": "2012-06-15 08:00:00", "season": 2, "holiday": 0, "workingday": 1,
       "weather": 1, "temp": 24.6, "humidity": 60, "windspeed": 12.998}'
```

Or pull the image CI already built (replace `<sha>` with a commit SHA from `main`):

```bash
docker run --rm -p 8000:8000 ghcr.io/rr3-10/bike-demand-mlops:<sha>
```

## API

| Endpoint | What it does |
|---|---|
| `GET /health` | `{"status": "ok", "model_version": "<git sha>"}`, so you can prove which version is running |
| `POST /predict` | Takes one hour of conditions, returns `{"predicted_count": 615, "model_version": "<git sha>"}` |

Input is validated with pydantic: `season` and `weather` must be 1–4, `holiday` and `workingday`
0 or 1, `humidity` 0–100, and unknown or renamed fields are rejected with a 422.

## Run without Docker (development)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train.train          # writes models/model.joblib
pytest -v                          # 8 tests
uvicorn src.app.main:app --port 8000
```

## Project layout

```
src/features.py        feature engineering shared by training and serving
src/train/train.py     trains the Random Forest, saves models/model.joblib
src/app/main.py        FastAPI app: /health and /predict
tests/                 pytest tests (API contract, model, schema, edge case, determinism)
Dockerfile             multi-stage, non-root, pinned base image, HEALTHCHECK
.github/workflows/ci.yml  lint + test on PRs; build, Trivy scan, push to GHCR on main
notebooks/             the original Kaggle notebook
```

## Model

| | |
|---|---|
| Algorithm | `RandomForestRegressor` (100 trees, max depth 20), trained on `log1p(count)` |
| Features | hour, day of week, weekend, rush hour, cyclical hour, season/weather one-hot, temp × humidity, workingday × hour |
| Dropped | `casual`, `registered` (they add up to the target: leakage), `atemp` (duplicate of `temp`) |
| Validation RMSLE | ≈ 0.40 on a 20% hold-out |

## CI/CD

- **Pull request to `main`:** ruff lint, train, pytest. `main` is protected, so a PR can only be merged when this check is green.
- **Push to `main`:** the same checks, then `docker build`, a Trivy scan (fails on fixable CRITICAL vulnerabilities), and a push to GHCR tagged with the commit SHA. The digest is shown in the run summary.
- Every third-party action is pinned to a full commit SHA.
