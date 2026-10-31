# ---------- Stage 1: builder (installs packages and trains the model) ----------
# Base image pinned to an exact version, so builds don't change under us.
FROM python:3.12.16-slim-trixie AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install packages into a virtual env we can copy to the final stage.
# requirements.txt is copied first so this layer is cached when only code changes.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# Train the model inside the build, so the image always carries a model
# made from this exact code and data.
COPY src/ src/
COPY data/ data/
RUN python -m src.train.train --data data/train.csv --out models


# ---------- Stage 2: runtime (small, no build tools, no training data) ----------
FROM python:3.12.16-slim-trixie AS runtime

# The git commit SHA, passed in by CI. Lets /health say which version is running.
ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION} \
    MODEL_PATH=/app/models/model.joblib \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Unprivileged user: if the app is ever broken into, the attacker is not root.
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/models ./models
COPY src/ src/

USER app
EXPOSE 8000

# Docker checks /health every 30s and marks the container unhealthy if it fails.
# slim images have no curl, so we use Python's built-in urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
