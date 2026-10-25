# Model Card - taxi-fare-predictor

_Generated: 2026-10-25 13:17 UTC_

## Model details

- **Problem:** fare
- **Algorithm:** LGBMRegressor
- **Target:** `fare_amount`
- **Registry version:** n/a

## Intended use

Operational forecasting of fare_amount to support fleet positioning and rider fare estimates. Not intended for individual pricing decisions or any use with legal/financial consequences for individuals.

## Evaluation - overall (test split)

| Metric | Value |
|--------|-------|
| rmse | 3.3046 |
| mae | 0.8143 |
| mape | 5.5093 |
| r2 | 0.9602 |

## Evaluation - subgroup fairness

### By borough

Disparity ratio (worst/best RMSE): **5.80** (threshold 1.5) - ⚠️ **flagged**. Worst: `Unknown`, best: `Manhattan`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| Unknown | 925 | 12.118 | 2.663 | 30.45 |
| Brooklyn | 1932 | 10.521 | 4.795 | 20.20 |
| Bronx | 466 | 9.609 | 5.861 | 19.18 |
| Queens | 23593 | 7.897 | 2.639 | 12.41 |
| Manhattan | 245373 | 2.088 | 0.590 | 4.60 |

### By time_of_day

Disparity ratio (worst/best RMSE): **1.83** (threshold 1.5) - ⚠️ **flagged**. Worst: `overnight`, best: `evening`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| overnight | 7020 | 5.548 | 1.365 | 6.18 |
| afternoon | 97932 | 3.465 | 0.880 | 7.40 |
| morning | 51445 | 3.175 | 0.872 | 4.62 |
| evening | 115897 | 3.028 | 0.700 | 4.26 |

### By day_type

Disparity ratio (worst/best RMSE): **1.00** (threshold 1.5) - ok. Worst: `weekday`, best: `weekday`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| weekday | 272294 | 3.305 | 0.814 | 5.51 |

### By holiday

Disparity ratio (worst/best RMSE): **1.00** (threshold 1.5) - ok. Worst: `non-holiday`, best: `non-holiday`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| non-holiday | 272294 | 3.305 | 0.814 | 5.51 |

### By airport

Disparity ratio (worst/best RMSE): **2.80** (threshold 1.5) - ⚠️ **flagged**. Worst: `airport`, best: `non-airport`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| airport | 20892 | 7.500 | 2.310 | 4.99 |
| non-airport | 251402 | 2.674 | 0.690 | 5.55 |

## Top features (SHAP)

| Feature | Mean |SHAP| |
|---------|-----------|
| trip_distance | 6.7039 |
| trip_duration_min | 3.9719 |
| PULocationID | 0.6328 |
| DOLocationID | 0.4711 |
| hour_cos | 0.1013 |
| passenger_count | 0.0856 |
| hour | 0.0698 |
| dow_sin | 0.0398 |
| hour_sin | 0.0350 |
| dow_cos | 0.0234 |

## Limitations

- Trained on a single month of NYC Yellow Taxi data; may not generalise to other periods, boroughs served sparsely, or green/for-hire vehicles.
- Predictions degrade under distribution shift (see drift monitoring).

## Ethical considerations

- No demographic or protected attributes are used or available; fairness is assessed as geographic/temporal service equity, not demographic parity.
- Large borough-level disparities could translate into unequal service quality across neighbourhoods and should be monitored.
