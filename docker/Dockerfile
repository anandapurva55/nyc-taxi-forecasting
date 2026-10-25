# Multi-stage image for the NYC Taxi prediction API.
# Build context is the repository root: docker build -f docker/Dockerfile .

# --- Builder: install dependencies into a venv --------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# --- Runtime: copy venv + source ---------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is required at runtime by XGBoost / LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY config ./config
COPY scripts ./scripts

# --- Baked model registry, artifacts and serving data ------------------------
# The API is self-contained: it loads models from the MLflow registry and reads
# the demand history at startup with no external mounts. `mlflow.container.db`
# is produced by scripts/make_mlflow_portable.py so artifact paths resolve at
# /app/mlruns inside this Linux image (the dev mlflow.db has absolute host paths).
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
COPY mlruns ./mlruns
COPY mlflow.container.db ./mlflow.db
COPY data/processed/demand_hourly.parquet ./data/processed/demand_hourly.parquet

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
