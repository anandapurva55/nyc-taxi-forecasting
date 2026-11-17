# API Examples

Set the base URL first. Locally it is the Uvicorn address. On AWS it is the
Terraform `api_url` output, an HTTPS address served by CloudFront. Plain HTTP is
refused there, and the load balancer behind it accepts traffic only from
CloudFront.

```bash
API_URL=http://127.0.0.1:8000
# AWS: API_URL=$(terraform -chdir=infra/terraform output -raw api_url)
```

Swagger UI is at `$API_URL/docs` (use **Authorize** to enter the API key), the
OpenAPI schema at `$API_URL/openapi.json` and Prometheus metrics at
`$API_URL/metrics`.

## API key

When the service runs with an `API_KEY` environment variable, every prediction,
forecast, explanation, `/version` and `/metrics` request must send it in the
`X-API-Key` header. Missing or wrong keys return HTTP 401. `/health` stays open
for load balancer checks, and `/docs` shows only the request schema. On AWS the
key is stored in AWS Systems Manager Parameter Store as a `SecureString` and
injected into the task by ECS.

```bash
KEY="X-API-Key: $API_KEY"
```

Locally without `API_KEY`, the header is ignored and can be left out. When
`RATE_LIMIT_PER_MINUTE` is set, a client that sends more requests than that
within one minute receives HTTP 429 with a `Retry-After` header. The client is
identified by the address the outermost trusted proxy recorded
(`TRUSTED_PROXY_HOPS`), so sending a made-up `X-Forwarded-For` header does not
reset the budget.

## Readiness and version

```bash
curl "$API_URL/health"
curl -H "$KEY" "$API_URL/version"
```

The service is ready only when the body shows `"status": "ok"`, the model flags
are `true`, and `history_hours` and `zones` are above zero. The endpoint returns
HTTP 200 with a `degraded` status when a dependency is missing, so always read
the body.

`/version` reports the commit the image was built from, the exact registered
version and MLflow run of every served model, SHA-256 hashes of the baked model
registry and demand history, how fares are quoted without a trip duration, and
how many hours ahead the demand model serves.

## Fare prediction and explanation

```bash
FARE='{
  "trip_distance": 3.4,
  "pu_location_id": 132,
  "do_location_id": 230,
  "passenger_count": 1,
  "pickup_datetime": "2024-02-27T18:30:00"
}'
curl -X POST "$API_URL/predict/fare" -H "$KEY" -H "Content-Type: application/json" -d "$FARE"
curl -X POST "$API_URL/explain/fare" -H "$KEY" -H "Content-Type: application/json" -d "$FARE"
```

The response predicts `fare_amount` in dollars, not the full `total_amount`
bill, and `model` names the registered model that answered:

- Without `trip_duration_min`, as when a fare is quoted before the trip, the
  booking-time model `taxi-fare-booking` answers. It was trained without the
  duration.
- With `trip_duration_min`, the fare model `taxi-fare-predictor` answers.

`low_confidence` is `true`, with a reason in `notes`, when the pickup or
drop-off zone is unknown (zones 264 and 265), the group with the largest errors
in the subgroup audit. SHAP contributions show how the model combined the
inputs. They do not describe cause and effect.

## Multi-hour demand forecast

```bash
curl -X POST "$API_URL/forecast/demand" -H "$KEY" -H "Content-Type: application/json" \
  -d '{"horizon_hours": 4, "zones": [132, 161, 230]}'
```

The forecast starts at the hour after the last stored history hour
(2024-02-29 23:00 for the February 2024 model). Each point carries a `method`:

- `model` for the first `model_horizon_hours` (2). The model forecasts
  recursively, so each forecast hour becomes the lag input for the next hour.
- `same_hour_last_week` after that: the count at the same hour one week
  earlier. In the rolling backtest this rule was more accurate than the model
  from three hours ahead.

## Single demand prediction and explanation

```bash
DEMAND='{"zone": 132, "target_hour": "2024-03-01T00:00:00"}'
curl -X POST "$API_URL/predict/demand" -H "$KEY" -H "Content-Type: application/json" -d "$DEMAND"
curl -X POST "$API_URL/explain/demand" -H "$KEY" -H "Content-Type: application/json" -d "$DEMAND"
```

Single predictions follow the same routing and report their `method`.
Explanations cover hours up to one hour past the latest observed hour, where
every model input is observed; later hours return HTTP 400.

## Showcase request

JFK Airport (zone 132) to Times Square (zone 230), 17 miles, one passenger, on
27 February 2024 at 18:30:

```bash
curl -X POST "$API_URL/predict/fare" -H "$KEY" -H "Content-Type: application/json" -d '{
  "trip_distance": 17.0,
  "pu_location_id": 132,
  "do_location_id": 230,
  "passenger_count": 1,
  "pickup_datetime": "2024-02-27T18:30:00"
}'
```

## Validation errors

Requests that break the schema return HTTP 422 with the field and the rule that
failed:

```bash
curl -i -X POST "$API_URL/predict/fare" -H "$KEY" -H "Content-Type: application/json" -d '{
  "trip_distance": -5,
  "pu_location_id": 132,
  "do_location_id": 230,
  "pickup_datetime": "2024-02-27T18:30:00"
}'
```

## Full contract check

[scripts/aws/verify_staging.py](../scripts/aws/verify_staging.py) calls every
endpoint, including an invalid request, and exits with code 0 only when all
checks pass. It reads the key from the `API_KEY` environment variable:

```bash
API_KEY=... .venv/bin/python scripts/aws/verify_staging.py --api-url "$API_URL" --timeout 60
```
