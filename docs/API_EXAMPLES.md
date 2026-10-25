# API Examples

Set the base URL first. Locally it is the Uvicorn address. On AWS it is the
Terraform `api_url` output.

```bash
API_URL=http://127.0.0.1:8000
# AWS: API_URL=$(terraform -chdir=infra/terraform output -raw api_url)
```

Swagger UI is at `$API_URL/docs`, the OpenAPI schema at `$API_URL/openapi.json`
and Prometheus metrics at `$API_URL/metrics`.

## Readiness

```bash
curl "$API_URL/health"
```

The service is ready only when the body shows `"status": "ok"`, both
`demand_model_loaded` and `fare_model_loaded` are `true`, and `history_hours`
and `zones` are above zero. The endpoint returns HTTP 200 with a `degraded`
status when a dependency is missing, so always read the body.

## Fare prediction and explanation

```bash
FARE='{
  "trip_distance": 3.4,
  "pu_location_id": 132,
  "do_location_id": 230,
  "passenger_count": 1,
  "pickup_datetime": "2024-02-27T18:30:00"
}'
curl -X POST "$API_URL/predict/fare" -H "Content-Type: application/json" -d "$FARE"
curl -X POST "$API_URL/explain/fare" -H "Content-Type: application/json" -d "$FARE"
```

The response predicts `fare_amount` in dollars, not the full `total_amount`
bill. When `trip_duration_min` is left out, the service estimates it and sets
`duration_estimated` to `true`. SHAP contributions show how the model combined
the inputs. They do not describe cause and effect.

## Recursive demand forecast

```bash
curl -X POST "$API_URL/forecast/demand" -H "Content-Type: application/json" \
  -d '{"horizon_hours": 3, "zones": [132, 161, 230]}'
```

The forecast starts at the hour after the last stored history hour
(2024-02-29 23:00 for the February 2024 model). Each forecast hour becomes the
lag input for the next hour.

## Single demand prediction and explanation

```bash
DEMAND='{"zone": 132, "target_hour": "2024-03-01T00:00:00"}'
curl -X POST "$API_URL/predict/demand" -H "Content-Type: application/json" -d "$DEMAND"
curl -X POST "$API_URL/explain/demand" -H "Content-Type: application/json" -d "$DEMAND"
```

## Showcase request

JFK Airport (zone 132) to Times Square (zone 230), 17 miles, one passenger, on
27 February 2024 at 18:30:

```bash
curl -X POST "$API_URL/predict/fare" -H "Content-Type: application/json" -d '{
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
curl -i -X POST "$API_URL/predict/fare" -H "Content-Type: application/json" -d '{
  "trip_distance": -5,
  "pu_location_id": 132,
  "do_location_id": 230,
  "pickup_datetime": "2024-02-27T18:30:00"
}'
```

## Full contract check

[scripts/aws/verify_staging.py](../scripts/aws/verify_staging.py) calls every
endpoint, including an invalid request, and exits with code 0 only when all
checks pass:

```bash
.venv/bin/python scripts/aws/verify_staging.py --api-url "$API_URL" --timeout 60
```
