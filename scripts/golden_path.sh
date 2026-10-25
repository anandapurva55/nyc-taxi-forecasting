#!/usr/bin/env bash
# Golden path: run the whole project from a clean checkout and verify every stage.
#
# Stages: environment, quality gate, data pipeline, training and registration,
# batch inference, live API smoke test, responsible AI analysis, drift report.
# Each stage writes its log to logs/golden_path/ and the run stops at the first
# failure. A JSON summary is written to logs/golden_path/summary.json.
#
# Usage (from the repository root):
#   bash scripts/golden_path.sh                 # February 2024, 5 tuning trials
#   YEAR=2024 MONTH=2 TRIALS=5 bash scripts/golden_path.sh
#   CLEAN=1 bash scripts/golden_path.sh         # delete generated outputs first

set -euo pipefail

YEAR="${YEAR:-2024}"
MONTH="${MONTH:-2}"
TRIALS="${TRIALS:-5}"
API_PORT="${API_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs/golden_path"
mkdir -p "$LOG_DIR"
PY=".venv/bin/python"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAGES=()
API_PID=""

cleanup() {
    if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID"
        wait "$API_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

stage() {
    local name="$1"
    shift
    local log="$LOG_DIR/${name}.log"
    local t0=$SECONDS
    printf '==> %-22s ' "$name"
    if "$@" >"$log" 2>&1; then
        local secs=$((SECONDS - t0))
        printf 'ok (%ss)\n' "$secs"
        STAGES+=("{\"stage\": \"$name\", \"status\": \"ok\", \"seconds\": $secs, \"log\": \"$log\"}")
    else
        printf 'FAILED, see %s\n' "$log"
        tail -n 20 "$log"
        exit 1
    fi
}

require_file() {
    for f in "$@"; do
        [[ -s "$f" ]] || { echo "missing output: $f"; return 1; }
        echo "found $f"
    done
}

setup_env() {
    [[ -x "$PY" ]] || "$PYTHON_BIN" -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
    "$PY" --version
}

serve_and_smoke_test() {
    "$PY" -m uvicorn src.serving.api:app --host 127.0.0.1 --port "$API_PORT" \
        >"$LOG_DIR/api_server.log" 2>&1 &
    API_PID=$!
    for _ in $(seq 1 60); do
        curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 && break
        sleep 2
    done
    curl -fsS "http://127.0.0.1:$API_PORT/health"
    echo
    "$PY" scripts/aws/verify_staging.py --api-url "http://127.0.0.1:$API_PORT" \
        --timeout 60 --output "$LOG_DIR/api_smoke_test.json"
    kill "$API_PID"
    wait "$API_PID" 2>/dev/null || true
    API_PID=""
}

if [[ "${CLEAN:-0}" == "1" ]]; then
    echo "==> removing generated outputs"
    rm -rf data/raw/* data/processed/* data/features/* mlruns mlflow.db \
        predictions.db artifacts docs/model_cards/demand.md docs/model_cards/fare.md
fi

stage 01_environment setup_env
stage 02_lint "$PY" -m ruff check .
stage 03_tests "$PY" -m pytest -q
stage 04_data_pipeline "$PY" -m scripts.run_pipeline --year "$YEAR" --month "$MONTH"
stage 05_pipeline_outputs require_file \
    data/processed/pipeline_summary.json data/processed/demand_hourly.parquet \
    data/features/demand_train.parquet data/features/demand_val.parquet data/features/demand_test.parquet \
    data/features/fare_train.parquet data/features/fare_val.parquet data/features/fare_test.parquet
stage 06_train_and_register "$PY" -m scripts.train_models --n-trials "$TRIALS"
stage 07_batch_inference "$PY" -m scripts.batch_forecast --horizon 3
stage 08_api_smoke_test serve_and_smoke_test
stage 09_responsible_ai "$PY" -m scripts.responsible.run_analysis
stage 10_drift_report "$PY" -m scripts.monitoring.demo_drift_report --problem all --mode drift
stage 11_final_outputs require_file \
    mlflow.db predictions.db docs/model_cards/demand.md docs/model_cards/fare.md \
    artifacts/responsible/demand_fairness.json artifacts/responsible/fare_fairness.json \
    artifacts/drift/demand_drift_metrics.json artifacts/drift/fare_drift_metrics.json

{
    echo "{"
    echo "  \"started_at\": \"$STARTED_AT\","
    echo "  \"finished_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"git_commit\": \"$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\","
    echo "  \"data_month\": \"$YEAR-$(printf '%02d' "$MONTH")\","
    echo "  \"tuning_trials\": $TRIALS,"
    echo "  \"stages\": ["
    for i in "${!STAGES[@]}"; do
        sep=","
        [[ $i -eq $((${#STAGES[@]} - 1)) ]] && sep=""
        echo "    ${STAGES[$i]}$sep"
    done
    echo "  ]"
    echo "}"
} >"$LOG_DIR/summary.json"

echo "==> all stages passed. Summary: $LOG_DIR/summary.json"
