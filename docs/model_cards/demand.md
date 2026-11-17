# Model Card - taxi-demand-forecaster

_Generated: 2026-11-17 09:06 UTC_

## Model details

- **Problem:** demand
- **Algorithm:** XGBRegressor
- **Target:** `pickup_count`
- **Registry version:** 2 (@champion)

## Intended use

Operational forecasting of pickup_count to support fleet positioning and rider fare estimates. Not intended for individual pricing decisions or any use with legal/financial consequences for individuals.

## Evaluation - overall (test split)

| Metric | Value |
|--------|-------|
| rmse | 11.3641 |
| mae | 3.4865 |
| mape | 43.1672 |
| r2 | 0.9679 |

## Evaluation - subgroup fairness

### By borough

Disparity ratio (worst/best WAPE): **1.80** (threshold 1.5) - ⚠️ **flagged**. RMSE ratio for context: 5.66. Worst: `Unknown`, best: `Manhattan`. Not compared (mean actual below 1, too small for a stable WAPE): `Staten Island`, `Bronx`, `EWR`, `Brooklyn`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| Staten Island | 634 | 0.00 | 32.102 | 0.126 | 0.101 |
| Bronx | 2268 | 0.28 | 1.286 | 0.586 | 0.363 |
| EWR | 53 | 0.43 | 1.201 | 0.753 | 0.521 |
| Brooklyn | 3224 | 0.81 | 0.851 | 1.108 | 0.687 |
| Unknown | 106 | 9.39 | 0.273 | 3.747 | 2.566 |
| Queens | 3647 | 6.23 | 0.230 | 6.599 | 1.432 |
| Manhattan | 3480 | 72.37 | 0.151 | 21.220 | 10.959 |

### By time_of_day

Disparity ratio (worst/best WAPE): **3.05** (threshold 1.5) - ⚠️ **flagged**. RMSE ratio for context: 5.41. Worst: `overnight`, best: `afternoon`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| overnight | 3048 | 2.63 | 0.391 | 3.096 | 1.028 |
| evening | 4268 | 29.75 | 0.182 | 16.740 | 5.406 |
| morning | 3048 | 19.50 | 0.164 | 7.681 | 3.193 |
| afternoon | 3048 | 27.70 | 0.128 | 10.358 | 3.551 |

### By day_type

Coverage gap: fewer than two groups with enough volume appear in the evaluation data, so this dimension could not be audited.

### By holiday

Coverage gap: fewer than two groups with enough volume appear in the evaluation data, so this dimension could not be audited.

### By airport

Disparity ratio (worst/best WAPE): **1.17** (threshold 1.5) - ok. RMSE ratio for context: 3.43. Worst: `non-airport`, best: `airport`.

| Group | n | Mean actual | WAPE | RMSE | MAE |
|-------|---|-------------|------|------|-----|
| non-airport | 13307 | 19.49 | 0.169 | 10.913 | 3.302 |
| airport | 105 | 186.04 | 0.145 | 37.452 | 26.917 |

## Top features (SHAP)

| Feature | Mean absolute SHAP value |
|---------|-----------|
| lag_168 | 13.3500 |
| lag_1 | 11.9386 |
| lag_24 | 3.3996 |
| rolling_mean_3 | 2.1767 |
| lag_3 | 0.8099 |
| hour_cos | 0.7388 |
| lag_2 | 0.4341 |
| hour | 0.3924 |
| rolling_std_3 | 0.3200 |
| rolling_std_168 | 0.3061 |

## Limitations

- Trained on a single month of NYC Yellow Taxi data; may not generalise to other periods, boroughs served sparsely, or green/for-hire vehicles.
- Predictions degrade under distribution shift (see drift monitoring).

## Ethical considerations

- No demographic or protected attributes are used or available; fairness is assessed as geographic/temporal service equity, not demographic parity.
- Large borough-level disparities could translate into unequal service quality across neighbourhoods and should be monitored.
