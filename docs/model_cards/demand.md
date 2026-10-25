# Model Card - taxi-demand-forecaster

_Generated: 2026-10-25 13:17 UTC_

## Model details

- **Problem:** demand
- **Algorithm:** XGBRegressor
- **Target:** `pickup_count`
- **Registry version:** n/a

## Intended use

Operational forecasting of pickup_count to support fleet positioning and rider fare estimates. Not intended for individual pricing decisions or any use with legal/financial consequences for individuals.

## Evaluation - overall (test split)

| Metric | Value |
|--------|-------|
| rmse | 9.3607 |
| mae | 2.8675 |
| mape | 42.3865 |
| r2 | 0.9723 |

## Evaluation - subgroup fairness

### By borough

Disparity ratio (worst/best RMSE): **351.78** (threshold 1.5) - ⚠️ **flagged**. Worst: `Manhattan`, best: `Staten Island`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| Manhattan | 3485 | 17.130 | 8.896 | 31.99 |
| Queens | 3645 | 6.178 | 1.271 | 53.48 |
| Unknown | 104 | 3.655 | 2.218 | 36.37 |
| Brooklyn | 3221 | 0.850 | 0.514 | 55.71 |
| Bronx | 2272 | 0.468 | 0.272 | 62.82 |
| EWR | 53 | 0.234 | 0.116 | 92.14 |
| Staten Island | 526 | 0.049 | 0.022 | 91.86 |

### By time_of_day

Disparity ratio (worst/best RMSE): **4.56** (threshold 1.5) - ⚠️ **flagged**. Worst: `evening`, best: `overnight`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| evening | 4234 | 13.001 | 4.124 | 39.23 |
| afternoon | 3024 | 9.648 | 3.281 | 38.78 |
| morning | 3024 | 6.904 | 2.683 | 41.15 |
| overnight | 3024 | 2.854 | 0.880 | 56.44 |

### By day_type

Disparity ratio (worst/best RMSE): **1.00** (threshold 1.5) - ok. Worst: `weekday`, best: `weekday`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| weekday | 13306 | 9.361 | 2.868 | 42.39 |

### By holiday

Disparity ratio (worst/best RMSE): **1.00** (threshold 1.5) - ok. Worst: `non-holiday`, best: `non-holiday`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| non-holiday | 13306 | 9.361 | 2.868 | 42.39 |

### By airport

Disparity ratio (worst/best RMSE): **3.96** (threshold 1.5) - ⚠️ **flagged**. Worst: `airport`, best: `non-airport`.

| Group | n | RMSE | MAE | MAPE |
|-------|---|------|-----|------|
| airport | 106 | 35.098 | 26.218 | 27.93 |
| non-airport | 13200 | 8.856 | 2.680 | 42.67 |

## Top features (SHAP)

| Feature | Mean |SHAP| |
|---------|-----------|
| lag_168 | 13.3060 |
| lag_1 | 5.9565 |
| lag_24 | 4.8771 |
| rolling_mean_3 | 2.5716 |
| lag_3 | 0.6576 |
| hour_cos | 0.5167 |
| rolling_mean_168 | 0.4461 |
| hour | 0.4119 |
| rolling_std_168 | 0.3354 |
| rolling_std_3 | 0.3305 |

## Limitations

- Trained on a single month of NYC Yellow Taxi data; may not generalise to other periods, boroughs served sparsely, or green/for-hire vehicles.
- Predictions degrade under distribution shift (see drift monitoring).

## Ethical considerations

- No demographic or protected attributes are used or available; fairness is assessed as geographic/temporal service equity, not demographic parity.
- Large borough-level disparities could translate into unequal service quality across neighbourhoods and should be monitored.
