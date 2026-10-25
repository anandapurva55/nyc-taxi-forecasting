# Data Scope and Model Limitations

## Source

The default input is NYC TLC Yellow Taxi trip data for February 2024 plus the taxi
zone lookup, obtained from the [official trip-record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Review the provider's data dictionary, publication notices, terms and corrections
before redistribution or use. No source data or trained model is bundled here.

## Targets and Availability

Fare predicts `fare_amount`, not tips, tolls or the full `total_amount`. Actual
trip duration is a training feature; the API estimates duration when it is
omitted. That approximation creates a training-versus-serving difference.
Features must be available at the intended prediction time. Weather and traffic
feeds are not current model inputs.

Demand predicts cleaned pickup counts for a zone/hour. Zero counts are inserted
to form a dense grid. Cleaning removes invalid/out-of-period rows according to
configured bounds. Review the generated cleaning report to understand exclusions.
Time splits are chronological row slices and can share boundary timestamps;
they are not necessarily separate whole-day or whole-month partitions.

At serving time, missing prior demand hours are also filled with zero counts.
The API does not distinguish these gaps from observed zero-pickup hours or
reject a request solely because its history window is incomplete. Keep saved
history current and check its coverage; HTTP success is not a data-quality check.

## Evaluation Limits

Historical held-out performance estimates behavior only within the evaluated
data scope. It does not demonstrate generalization to new cities, future fare
policies, weather events or major demand changes. MAE/RMSE measure error magnitude;
direction needs signed residuals. MAPE is sensitive to small targets. R-squared
is not a percentage of predictions that are correct.

Subgroup analysis compares borough, time, day type, holiday and airport errors.
Review sample count, target scale, coverage and unknown location groups before
interpreting a disparity. A single observed group does not prove parity. A flag
locates an error difference; it does not establish its cause or approve a remedy.

SHAP describes the fitted model's contributions, not causal effects or fairness.
Generated cards may show unresolved model versions. Serving and batch references
such as `latest` are mutable, so record a resolved model identity separately when
auditable lineage is required.

This application does not implement an official fare quote, pricing policy,
automated dispatch policy or fraud decision system. Do not treat dashboard KPIs
as measured business benefits or realized revenue. Operational use requires
current data, monitoring, appropriate review and human decision ownership.