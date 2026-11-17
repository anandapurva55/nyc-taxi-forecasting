# Multi-stage image for the NYC Taxi prediction API.
# Build context is the repository root: docker build -f docker/Dockerfile .

# --- Builder: install dependencies into a venv --------------------------------
# The official Python image through Amazon ECR Public, which AWS builds can pull
# without Docker Hub's anonymous rate limit.
FROM public.ecr.aws/docker/library/python:3.15-slim AS builder

# uv installs the exact versions in uv.lock (hash-checked) into /opt/venv with
# the image's own Python. --frozen fails the build if the lock is out of date.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Runtime dependencies only: no notebook, test, lint or AWS tooling.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Runtime: copy venv + source ---------------------------------------------
FROM public.ecr.aws/docker/library/python:3.15-slim AS runtime

# libgomp1 is required at runtime by XGBoost / LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

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
# config/build_info.json (written by scripts/write_build_info.py) records the
# commit, model versions and input hashes; the API serves it at /version.
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
COPY mlruns ./mlruns
COPY mlflow.container.db ./mlflow.db
COPY data/processed/demand_hourly.parquet ./data/processed/demand_hourly.parquet

# Run as an unprivileged user. MLflow opens the SQLite registry read-write, so
# the app user owns /app.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
