# Model Card - taxi-fare-booking

_Generated: 2026-11-17 09:06 UTC_

## Model details

- **Problem:** fare_booking
- **Algorithm:** LGBMRegressor
- **Target:** `fare_amount`
- **Registry version:** 1 (@champion)

## Intended use

Operational forecasting of fare_amount to support fleet positioning and rider fare estimates. Not intended for individual pricing decisions or any use with legal/financial consequences for individuals.

## Evaluation - overall (test split)

| Metric | Value |
|--------|-------|
| rmse | 3.9771 |
| mae | 1.8156 |
| mape | 12.8490 |
| r2 | 0.9424 |

## Evaluation - subgroup fairness

### By borough

Disparity ratio (worst/best WAPE): **2.98** (threshold 1.5) - ⚠️ **flagged**. RMSE ratio for context: 3.71. Worst: `Bronx`, best: `Queens`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| Bronx | 466 | 35.52 | 0.182 | 10.876 | 6.473 |
| Brooklyn | 1932 | 28.51 | 0.177 | 10.594 | 5.040 |
| Unknown | 925 | 20.83 | 0.172 | 9.997 | 3.581 |
| Manhattan | 245373 | 15.14 | 0.108 | 2.935 | 1.639 |
| Queens | 23593 | 52.62 | 0.061 | 8.710 | 3.216 |

### By time_of_day

Disparity ratio (worst/best WAPE): **1.39** (threshold 1.5) - ok. RMSE ratio for context: 1.61. Worst: `morning`, best: `overnight`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| morning | 51445 | 18.03 | 0.110 | 3.840 | 1.986 |
| afternoon | 97932 | 19.52 | 0.109 | 4.274 | 2.128 |
| evening | 115897 | 17.66 | 0.084 | 3.621 | 1.476 |
| overnight | 7020 | 22.94 | 0.079 | 5.845 | 1.819 |

### By day_type

Coverage gap: fewer than two groups with enough volume appear in the evaluation data, so this dimension could not be audited.

### By holiday

Coverage gap: fewer than two groups with enough volume appear in the evaluation data, so this dimension could not be audited.

### By airport

Disparity ratio (worst/best WAPE): **2.15** (threshold 1.5) - ⚠️ **flagged**. RMSE ratio for context: 2.45. Worst: `non-airport`, best: `airport`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| non-airport | 251402 | 15.56 | 0.111 | 3.381 | 1.732 |
| airport | 20892 | 54.38 | 0.052 | 8.281 | 2.821 |

## Top features (SHAP)

| Feature | Mean |SHAP| |
|---------|-----------|
| trip_distance | 10.3027 |
| hour_cos | 0.6765 |
| DOLocationID | 0.6592 |
| PULocationID | 0.5002 |
| airport_trip | 0.4677 |
| hour | 0.2463 |
| dow_cos | 0.2220 |
| dow_sin | 0.1535 |
| jfk_manhattan | 0.1526 |
| day_of_week | 0.1437 |

## Limitations

- Trained on a single month of NYC Yellow Taxi data; may not generalise to other periods, boroughs served sparsely, or green/for-hire vehicles.
- Predictions degrade under distribution shift (see drift monitoring).

## Ethical considerations

- No demographic or protected attributes are used or available; fairness is assessed as geographic/temporal service equity, not demographic parity.
- Large borough-level disparities could translate into unequal service quality across neighbourhoods and should be monitored.
