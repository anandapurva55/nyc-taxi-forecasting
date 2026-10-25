# Cost Optimization

This document summarises the cost profile of the taxi-forecasting platform on
AWS and the levers used (or available) to keep it lean. All figures are
**approximate, on-demand, us-east-1** estimates for illustration.

These are not a current pricing quote or a complete AWS bill. Confirm current
regional prices, public IPv4, data transfer, ALB capacity, monitoring, and any
out-of-band resources before budgeting. Actual sizing may differ from defaults.

## Cost profile

| Component | Driver | Indicative cost |
|-----------|--------|-----------------|
| API - ECS Fargate task | 0.25 vCPU / 0.5 GB, 1 task, 24×7 | ~$9–12 / mo |
| API - Application Load Balancer | 1 ALB, low LCU | ~$16–20 / mo |
| ECR storage | Last 5 images (lifecycle policy) | < $1 / mo |
| CloudWatch Logs | 14-day retention | low, usage-based |
| **API subtotal** | | **~$25–35 / mo** |
| Dashboard (opt-in) | 0.5 vCPU / 1 GB Fargate + its own ALB | **~$25–35 / mo** |
| Monitoring (Phase 5) | S3 + Firehose + CloudWatch + scheduled jobs + SNS | storage, delivery, metrics/alarms, and job usage |
| CI/CD (CodeBuild) | Per build-minute | pay-per-use |

This distribution defaults `enable_monitoring=false`; dashboard, CI/CD, data lake,
and Athena flags also default to `false`. To plan a new API-only deployment, explicitly
disable monitoring and keep other optional stacks disabled. Do not toggle flags
on an existing deployment without reviewing the plan for resource deletions.
Drift scheduling additionally requires `monitoring_image_uri`; retraining
scheduling requires `retrain_pipeline_arn`. Both schedules default to weekly.

Successful `terraform destroy` removes resources managed by that state, not
every resource in an account. Separately check running Processing jobs,
out-of-band tracking servers, retained storage, and other resources. Stop
schedules and jobs deliberately, and verify teardown and billing afterward.

## Optimization levers

### Compute (Fargate)
- **Right-sizing.** The API defaults to the smallest Fargate size
  (256 CPU / 512 MB); the dashboard uses 512/1024 because Streamlit is heavier.
  Sizes are Terraform variables (`task_cpu`/`task_memory`,
  `dashboard_task_cpu`/`dashboard_task_memory`) so they can be tuned to observed
  utilisation.
- **SageMaker job sizing.** Drift and retraining currently run as SageMaker
  Processing jobs, not Fargate tasks. Tune frequency, instance size, trial
  count, and runtime. Fargate Spot would require a separate ECS batch execution
  path and interruption handling; it is not a switch for the existing jobs.
- **Scale to zero when idle.** For demo/staging, set `desired_count = 0` (or
  destroy) outside working hours; scheduling this is a future addition, not
  implemented here. Zero tasks do not stop ALB, storage, or monitoring charges.

### Serving latency & throughput
- **Self-contained image.** Models, MLflow registry and demand history are baked
  in, so there are **no per-request database/object-store calls** - lower
  latency and fewer remote dependencies. Network and load-balancer charges can
  still apply.
- **Cached SHAP explainers.** Each explainer is built lazily on its first
  explanation request and cached in the process. Later requests reuse the
  explainer but still calculate SHAP values; cold requests may take longer.
- **In-process tree models.** XGBoost/LightGBM inference is CPU-cheap; no GPU is
  required.
- **Dashboard caching.** Streamlit `st.cache_data` (TTL) avoids repeat API calls
  for identical forecasts.

### Inference cost reduction (available)
- **Model compression** - the tree models are small; if size ever matters,
  prune/limit depth or quantise leaf values.
- **Batch over real-time** - precompute demand forecasts on a schedule and serve
  them from storage rather than recomputing the expensive recursive multi-hour,
  all-zone forecast per request (see roadmap).

### Storage
- **ECR lifecycle policy** keeps only the last 5 images per repo.
- **Log retention** capped at 14 days on the API/dashboard log groups.
- **Capture/drift data** in S3 expires after `capture_retention_days` (default
  90) via a lifecycle rule.
- **Partition by date/zone** for any future long-term forecast/history store to
  keep scans cheap.

### Data transfer
- Keeping the dashboard and API in the **same region** avoids cross-region
  transfer on that route. Same-region/VPC placement does not guarantee zero
  network cost; the dashboard uses the API ALB, and cross-AZ or public-address
  traffic may incur charges.

## Recommendations

1. For a new **core API-only** deployment, explicitly disable monitoring and
  leave other optional stacks disabled; review the full cost estimate first.
2. Right-size SageMaker Processing jobs and bound training trials and schedules.
3. Move the city-wide multi-hour demand view to **precomputed batch forecasts**
   to cut both latency and compute.
4. Review Fargate sizing against CloudWatch utilisation metrics after a
   representative load period and right-size down where possible.
