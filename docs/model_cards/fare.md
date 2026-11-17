# Model Card - taxi-fare-predictor

_Generated: 2026-11-17 09:06 UTC_

## Model details

- **Problem:** fare
- **Algorithm:** LGBMRegressor
- **Target:** `fare_amount`
- **Registry version:** 2 (@champion)

## Intended use

Operational forecasting of fare_amount to support fleet positioning and rider fare estimates. Not intended for individual pricing decisions or any use with legal/financial consequences for individuals.

## Evaluation - overall (test split)

| Metric | Value |
|--------|-------|
| rmse | 3.2130 |
| mae | 0.7741 |
| mape | 5.6568 |
| r2 | 0.9624 |

## Evaluation - subgroup fairness

### By borough

Disparity ratio (worst/best WAPE): **4.29** (threshold 1.5) - ⚠️ **flagged**. RMSE ratio for context: 5.00. Worst: `Brooklyn`, best: `Manhattan`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| Brooklyn | 1932 | 28.51 | 0.164 | 10.334 | 4.671 |
| Bronx | 466 | 35.52 | 0.161 | 9.480 | 5.710 |
| Unknown | 925 | 20.83 | 0.116 | 10.254 | 2.424 |
| Queens | 23593 | 52.62 | 0.044 | 7.653 | 2.320 |
| Manhattan | 245373 | 15.14 | 0.038 | 2.067 | 0.578 |

### By time_of_day

Disparity ratio (worst/best WAPE): **1.46** (threshold 1.5) - ok. RMSE ratio for context: 1.85. Worst: `overnight`, best: `evening`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| overnight | 7020 | 22.94 | 0.055 | 5.429 | 1.264 |
| morning | 51445 | 18.03 | 0.046 | 3.050 | 0.833 |
| afternoon | 97932 | 19.52 | 0.043 | 3.397 | 0.836 |
| evening | 115897 | 17.66 | 0.038 | 2.931 | 0.666 |

### By day_type

Coverage gap: fewer than two groups with enough volume appear in the evaluation data, so this dimension could not be audited.

### By holiday

Coverage gap: fewer than two groups with enough volume appear in the evaluation data, so this dimension could not be audited.

### By airport

Disparity ratio (worst/best WAPE): **1.21** (threshold 1.5) - ok. RMSE ratio for context: 2.76. Worst: `non-airport`, best: `airport`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| non-airport | 251402 | 15.56 | 0.043 | 2.617 | 0.676 |
| airport | 20892 | 54.38 | 0.036 | 7.220 | 1.954 |

## Top features (SHAP)

| Feature | Mean |SHAP| |
|---------|-----------|
| trip_distance | 7.3007 |
| trip_duration_min | 4.1644 |
| airport_trip | 0.5912 |
| DOLocationID | 0.2885 |
| jfk_manhattan | 0.1685 |
| PULocationID | 0.1288 |
| passenger_count | 0.0858 |
| hour_cos | 0.0638 |
| hour_sin | 0.0289 |
| dow_sin | 0.0255 |

## Limitations

- Trained on a single month of NYC Yellow Taxi data; may not generalise to other periods, boroughs served sparsely, or green/for-hire vehicles.
- Predictions degrade under distribution shift (see drift monitoring).

## Ethical considerations

- No demographic or protected attributes are used or available; fairness is assessed as geographic/temporal service equity, not demographic parity.
- Large borough-level disparities could translate into unequal service quality across neighbourhoods and should be monitored.
