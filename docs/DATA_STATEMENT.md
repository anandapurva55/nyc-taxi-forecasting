# Data Scope and Model Limitations

## Source

The default input is NYC TLC Yellow Taxi trip data for February 2024 plus the taxi
zone lookup, obtained from the [official trip-record page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Review the provider's data dictionary, publication notices, terms and corrections
before redistribution or use. No source data or trained model is bundled here.

## Targets and Availability

Fare predicts `fare_amount`, not tips, tolls or the full `total_amount`. The
fare model uses the actual trip duration, which is known only after the trip.
Quotes made before the trip use a separate booking-time model trained without
the duration, so no model receives an input it never saw in training. Weather
and traffic feeds are not current model inputs.

Demand predicts pickup counts for a zone and hour. Every trip with a valid
pickup time and zone in the target month counts as a pickup, including trips the
fare rules drop for an implausible fare or passenger count. Zero counts are
inserted to form a dense grid. Review the generated cleaning report to
understand exclusions.
Time splits are chronological row slices and can share boundary timestamps;
they are not necessarily separate whole-day or whole-month partitions.

At serving time, the model forecasts the first two hours after the stored
history, feeding each forecast back in as the next hour's lag. Later hours use
the count at the same hour one week earlier. Hours missing from the stored
history are treated as zero pickups, so keep the history current and check its
coverage; HTTP success is not a data-quality check.

## Evaluation Limits

Historical held-out performance estimates behavior only within the evaluated
data scope. It does not demonstrate generalization to new cities, future fare
policies, weather events or major demand changes. MAE/RMSE measure error magnitude;
direction needs signed residuals. MAPE is sensitive to small targets. R-squared
is not a percentage of predictions that are correct.

Subgroup analysis compares borough, time, day type, holiday and airport errors
with WAPE. Groups averaging less than one unit per row are listed but left out
of the ratio, because WAPE is unstable near zero. Review sample count, target
scale, coverage and unknown location groups before interpreting a disparity. A single observed group does not prove parity. A flag
locates an error difference; it does not establish its cause or approve a remedy.

SHAP describes the fitted model's contributions, not causal effects or fairness.
Serving loads the `champion` alias and resolves it to an exact registered
version, which `/version`, every captured prediction, every batch forecast row
and each model card record.

This application does not implement an official fare quote, pricing policy,
automated dispatch policy or fraud decision system. Do not treat dashboard KPIs
as measured business benefits or realized revenue. Operational use requires
current data, monitoring, appropriate review and human decision ownership.