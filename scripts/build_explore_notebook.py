"""Build the exploration notebook notebooks/01_explore_dataset.ipynb.

Run once with the project venv to (re)generate the notebook. Kept as a script so
the notebook content is reviewable in plain text and reproducible.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_explore_dataset.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(
    """
# Exploring the NYC Taxi Dataset: Source to Every Output

A detailed, visual tour of the data behind this project, from the official source
to every file the models train on.

We will trace one month of data through every Phase 1 transformation:

1. **Official source** and local raw files.
2. **Raw schema** and the meaning of one trip row.
3. **Validation and cleaning**, including every configured rule.
4. **Demand branch**: trips to zone-hour counts to model features.
5. **Fare branch**: cleaned trips to per-trip model features.
6. **Chronological train, validation and test splits**.
7. **Persisted artifacts** and the code that consumes them.

> Run each cell with **Shift + Enter**. Read each table from left to right and
> explain what changed before moving to the next stage.
"""
)

md(
    """
## 0. Setup

This points Python at the project so we can reuse its real data-loading code.
"""
)
code(
    """
import sys
from pathlib import Path

# Make the project importable no matter where Jupyter was launched from.
ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import get_config

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 120)
config = get_config()
print("Project root:", ROOT)
print("Configured period:", config.data["tlc_year"], config.data["tlc_month"])
"""
)

md(
    """
## 1. Official source and local landing files

The data is the **official NYC TLC (Taxi & Limousine Commission) Trip Record
Data** - free, public, and anonymized. The project downloads one month at a time
(configured as **February 2024**) plus a zone lookup table that translates numeric
location identifiers into borough, zone and service-zone labels.

| Source object | Official location | Local landing file | Format |
|---|---|---|---|
| Yellow Taxi trip records | TLC monthly trip-data CDN | `data/raw/yellow_tripdata_2024-01.parquet` | Parquet |
| Taxi zone lookup | TLC miscellaneous-data CDN | `data/raw/taxi_zone_lookup.csv` | CSV |

The exact year, month, URLs and local directories come from `config/config.yaml`.
`src/data/ingestion.py` streams downloads in 1 MiB chunks, writes a temporary
`.part` file, then renames it only after completion. Existing files are cached
unless overwrite is requested.

### Who creates and updates it, and when?

- Authorized technology providers under the Taxi & Livery Passenger
    Enhancement Programs (TPEP/LPEP) collect and submit Yellow/Green trip records
    to NYC TLC. TLC states that it did not create those submitted records and does
    not guarantee their accuracy.
- **NYC TLC publishes the files monthly**, typically with an approximately
    **two-month delay** so vendors can complete submissions.
- There is no guaranteed calendar day. TLC can correct historical files and
    standardize schemas; check the official page's download links and Errata
    before treating a cached file as the latest version.
- This project does **not** poll for new months automatically. It downloads the
    year/month selected in configuration or `TLC_TRIP_YEAR` and
    `TLC_TRIP_MONTH`. The configured month is February 2024.

The official Yellow Taxi data dictionary documents the broader published
fields: pickup/drop-off times and zones, distance, rate and payment types,
passenger count, and itemized fare components. The project uses the eight-field
contract shown below.
"""
)
code(
    """
from src.data.ingestion import load_trips, load_zone_lookup, trip_filename

year = int(config.data["tlc_year"])
month = int(config.data["tlc_month"])
trip_name = trip_filename(year, month)
trip_url = f'{config.data["base_url"].rstrip("/")}/{trip_name}'

source_inventory = pd.DataFrame([
    {
        "object": "Yellow Taxi trips",
        "official_url": trip_url,
        "local_path": str(config.raw_dir / trip_name),
        "format": "Parquet",
    },
    {
        "object": "Taxi zones",
        "official_url": config.data["zone_lookup_url"],
        "local_path": str(config.raw_dir / "taxi_zone_lookup.csv"),
        "format": "CSV",
    },
])
source_inventory
"""
)

code(
    """
publication_responsibility = pd.DataFrame([
    ("Collection and provider delivery", "Authorized TPEP/LPEP technology providers", "Continuous operational submissions"),
    ("Public file publication", "NYC Taxi and Limousine Commission", "Monthly, typically about two months after the trip month"),
    ("Corrections/schema updates", "NYC Taxi and Limousine Commission", "As needed; review official Errata and metadata"),
    ("Project download selection", "Project maintainer", f"Configured snapshot: {year}-{month:02d}; not automatically advanced"),
], columns=["responsibility", "owner", "cadence"])
publication_responsibility
"""
)

md(
    """
### Load the raw inputs

The notebook expects the files to have been downloaded by
`python -m scripts.run_pipeline`, or cached by an earlier run. No data is
silently synthesized. **Each row in the trip Parquet is one completed trip.**
"""
)
code(
    """
trip_path = config.raw_dir / trip_name
zone_path = config.raw_dir / "taxi_zone_lookup.csv"

if not trip_path.exists():
    raise FileNotFoundError(
        f"Missing {trip_path}. Run: python -m scripts.run_pipeline"
    )

raw = load_trips(trip_path)
zones = load_zone_lookup(zone_path)
print(f"This single month contains {len(raw):,} trips and {raw.shape[1]} columns.")
raw.head()
"""
)

md(
    """
### Full raw schema, then the columns this project requires

TLC schemas can change over time, so first inspect every column actually present
in this file. The second table documents the eight columns required by this
project's validation contract.

| Column | Meaning |
|---|---|
| `tpep_pickup_datetime` | When the trip started |
| `tpep_dropoff_datetime` | When the trip ended |
| `passenger_count` | Number of riders |
| `trip_distance` | Distance in miles |
| `PULocationID` | **P**ick-**U**p zone number |
| `DOLocationID` | **D**rop-**O**ff zone number |
| `fare_amount` | Base fare (US$) - what we predict for *fare* |
| `total_amount` | Total incl. tip, tolls, fees |
"""
)
code(
    """
schema = pd.DataFrame({
    "column": raw.columns,
    "dtype": raw.dtypes.astype(str).values,
    "non_null": raw.notna().sum().values,
    "null_count": raw.isna().sum().values,
    "example": [raw[col].dropna().iloc[0] if raw[col].notna().any() else None for col in raw.columns],
})
schema
"""
)
code(
    """
used = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "PULocationID", "DOLocationID", "fare_amount", "total_amount",
]
raw[used].head(5)
"""
)

md(
    """
### Locations are zones, not GPS

To protect privacy, pickup/drop-off are given as one of **265 zone numbers**. The
lookup table translates a number into a borough + neighbourhood.
"""
)
code(
    """
print(f"{len(zones)} zones. A few examples:")
display(zones.head())
zones[zones["LocationID"].isin([4, 79, 132, 138, 230])]
"""
)

md(
    """
The lookup is descriptive metadata. `PULocationID` and `DOLocationID` remain the
model keys; joining on `LocationID` adds readable labels for analysis and the
dashboard. It does not reveal exact pickup or drop-off coordinates.
"""
)

md(
    """
## 2. Why does the data need cleaning?

Real-world data is messy. The raw file contains impossible records - negative
fares, zero passengers, 999-mile trips, even timestamps from the wrong year.
Let's *see* some of that mess.
"""
)
code(
    """
print("Fare amount - notice the negative minimum and huge maximum:")
print(raw["fare_amount"].describe()[["min", "max"]])
print("\\nA few clearly-bad rows (negative fares):")
raw.loc[raw["fare_amount"] < 0, used].head(3)
"""
)

md(
    """
The project's cleaning step validates the required schema, parses timestamps,
restricts rows to the configured month, derives trip duration, and applies the
bounds in `config/config.yaml`. The rules are displayed below before they are
applied.
"""
)
code(
    """
from src.data.validation import REQUIRED_COLUMNS, clean_trips

rules = pd.DataFrame([
    ("schema", "all required columns exist", ", ".join(REQUIRED_COLUMNS)),
    ("timestamps", "pickup and drop-off are non-null and parseable", "required"),
    ("location", "pickup and drop-off zone IDs are non-null", "required"),
    ("reporting period", "pickup timestamp belongs to configured year/month", f"{year}-{month:02d}"),
    ("trip duration", "minutes between drop-off and pickup", f'{config.cleaning["min_trip_duration_min"]} to {config.cleaning["max_trip_duration_min"]}'),
    ("trip distance", "recorded miles", f'{config.cleaning["min_trip_distance"]} to {config.cleaning["max_trip_distance"]}'),
    ("fare", "base fare in dollars", f'{config.cleaning["min_fare"]} to {config.cleaning["max_fare"]}'),
    ("passengers", "recorded passenger count; null is treated as 0", f'{config.cleaning["min_passenger_count"]} to {config.cleaning["max_passenger_count"]}'),
], columns=["rule", "meaning", "accepted value"])
rules
"""
)
code(
    """

clean, report = clean_trips(raw, config, year=year, month=month)
print(f"Before: {report.input_rows:,} trips")
print(f"After:  {report.output_rows:,} trips")
print(f"Dropped {report.total_dropped:,} bad rows. Breakdown:")
pd.Series(report.dropped).sort_values(ascending=False).to_frame("rows_dropped")
"""
)

md(
    """
## 3. Branch A: cleaned trips to hourly demand

Demand asks: **how many pickups will occur in a pickup zone during an hour?**
The aggregation keeps only pickup time and pickup zone, floors timestamps to the
hour, counts rows, then creates a complete zone-by-hour grid. Missing
combinations become zero pickups rather than disappearing.
"""
)
code(
    """
from src.data.aggregation import aggregate_hourly_demand

demand = aggregate_hourly_demand(clean)
print("Each row = one (zone, hour) with its pickup count:")
demand.head(5)
"""
)

code(
    """
print("Demand columns:", demand.columns.tolist())
print("Zones:", demand["PULocationID"].nunique())
print("Hours:", demand["pickup_hour"].nunique())
print("Rows (zones x hours):", len(demand))
print("Zero-pickup rows added/preserved:", int((demand["pickup_count"] == 0).sum()))
"""
)

md(
    """
**Chart: how does demand change across the day?** Averaged over all zones and
days, we can see the daily rhythm of the city.
"""
)
code(
    """
hourly = demand.assign(hour=demand["pickup_hour"].dt.hour).groupby("hour")["pickup_count"]
by_hour = hourly.mean()
by_hour_total = hourly.sum()
peak_hour = int(by_hour_total.idxmax())
print(f"Highest monthly total: hour {peak_hour:02d}:00 with {by_hour_total.max():,} accepted pickups")

plt.figure(figsize=(9, 4))
by_hour.plot(kind="bar", color="#2b8cbe")
plt.title("Average pickups per zone, by hour of day (NYC, Jan 2024)")
plt.xlabel("Hour of day")
plt.ylabel("Avg pickups")
plt.tight_layout()
plt.show()
"""
)

md(
    """
**Chart: which zones are busiest?** The top pickup zones are typically Midtown
Manhattan and the airports.
"""
)
code(
    """
top = (
    demand.groupby("PULocationID")["pickup_count"].sum()
    .sort_values(ascending=False).head(10)
    .rename_axis("LocationID").reset_index()
    .merge(zones[["LocationID", "Zone", "Borough"]], on="LocationID", how="left")
)
top["label"] = top["Zone"] + " (" + top["Borough"] + ")"

plt.figure(figsize=(9, 4))
plt.barh(top["label"][::-1], top["pickup_count"][::-1], color="#31a354")
plt.title("Top 10 busiest pickup zones (total Jan 2024 pickups)")
plt.xlabel("Total pickups")
plt.tight_layout()
plt.show()
top[["label", "pickup_count"]]
"""
)

md(
    """
## 4. Demand features: what the model sees

The target remains `pickup_count`. Calendar fields describe the prediction
hour. Lag fields provide earlier pickup counts from the **same zone**. Rolling
statistics summarize only earlier counts because the series is shifted by one
hour before each window is calculated. Rows without all configured lags are
dropped as a warm-up period.
"""
)
code(
    """
from src.features.demand_features import build_demand_features

demand_features = build_demand_features(demand, config)
print("Configured lags:", config.features["demand_lags"])
print("Configured rolling windows:", config.features["demand_rolling_windows"])
print("Shape:", demand_features.shape)
demand_features.head()
"""
)

md(
    """
## 5. Branch B: cleaned trips to fare features

Here, **each cleaned trip is one training example**. The model learns how
distance, passenger count, trip duration, pickup/drop-off zones, and pickup-time
calendar signals relate to the `fare_amount` target. Unlike demand, fare is not
aggregated: one cleaned trip remains one row.
"""
)
code(
    """
from src.features.fare_features import build_fare_features

fare_features = build_fare_features(clean, config)
print("Shape:", fare_features.shape)
fare_features.head()
"""
)

md(
    """
### Fare versus distance

Longer trips generally cost more, but the spread reminds us that distance alone
does not determine fare.

"""
)
code(
    """
sample = clean.sample(3000, random_state=0)

plt.figure(figsize=(7, 5))
plt.scatter(sample["trip_distance"], sample["fare_amount"], s=6, alpha=0.3, color="#e6550d")
plt.title("Fare vs. trip distance (3,000 sampled trips)")
plt.xlabel("Trip distance (miles)")
plt.ylabel("Fare amount ($)")
plt.tight_layout()
plt.show()
"""
)

md(
    """
## 6. Chronological train, validation and test splits

Both feature tables are sorted by time and sliced into contiguous 80%, 10% and
10% segments. Training sees the earliest rows, validation supports model
selection, and test estimates performance on the latest held-out rows. A random
split would allow future observations to leak into training.
"""
)
code(
    """
from src.data.splits import time_based_split

def split_inventory(name, frame, time_col):
    parts = time_based_split(frame, time_col, config)
    return pd.DataFrame([
        {
            "problem": name,
            "split": split_name,
            "rows": len(part),
            "first_time": part[time_col].min(),
            "last_time": part[time_col].max(),
        }
        for split_name, part in zip(("train", "validation", "test"), parts)
    ])

split_evidence = pd.concat([
    split_inventory("demand", demand_features, "pickup_hour"),
    split_inventory("fare", fare_features, "tpep_pickup_datetime"),
], ignore_index=True)
split_evidence
"""
)

md(
    """
## 7. Persisted artifacts and downstream consumers

`scripts/run_pipeline.py` executes the same functions and writes these outputs.
The notebook calculates stages in memory for inspection; the pipeline is the
owner of persisted artifacts.

| Stage | Persisted artifact | Row meaning | Main consumer |
|---|---|---|---|
| Raw trips | `data/raw/yellow_tripdata_2024-01.parquet` | One published TLC trip | validation |
| Zone lookup | `data/raw/taxi_zone_lookup.csv` | One TLC zone | analysis/dashboard labels |
| Clean trips | `data/processed/trips_clean.parquet` | One valid trip | fare features + demand aggregation |
| Hourly demand | `data/processed/demand_hourly.parquet` | One pickup zone-hour | demand features + serving history |
| Demand features | `data/features/demand_features.parquet` | One model-ready zone-hour | inspection/splitting |
| Fare features | `data/features/fare_features.parquet` | One model-ready trip | inspection/splitting |
| Demand splits | `data/features/demand_{train,val,test}.parquet` | Chronological model partitions | demand training/evaluation |
| Fare splits | `data/features/fare_{train,val,test}.parquet` | Chronological model partitions | fare training/evaluation |
| Pipeline summary | `data/processed/pipeline_summary.json` | Counts and cleaning evidence | audit/readiness checks |

The demand serving service also reads `data/processed/demand_hourly.parquet` as
recent history for lag and rolling features. Model training reads the split
Parquet files. Human-readable zone names are joined later for reporting; they
are not personal location traces.
"""
)
code(
    """
artifact_inventory = [
    config.raw_dir / trip_name,
    config.raw_dir / "taxi_zone_lookup.csv",
    config.processed_dir / "trips_clean.parquet",
    config.processed_dir / "demand_hourly.parquet",
    config.processed_dir / "pipeline_summary.json",
    config.features_dir / "demand_features.parquet",
    config.features_dir / "fare_features.parquet",
] + [
    config.features_dir / f"{problem}_{split}.parquet"
    for problem in ("demand", "fare")
    for split in ("train", "val", "test")
]

pd.DataFrame({
    "path": [str(path.relative_to(ROOT)) for path in artifact_inventory],
    "exists_after_pipeline_run": [path.exists() for path in artifact_inventory],
})
"""
)

md(
    """
## 8. Lineage checks and limitations

Use these checks to defend the lineage rather than merely showing files:

1. Clean rows must not exceed raw rows; the cleaning report explains removals.
2. Demand has exactly one row for every observed pickup-zone/hour combination
   after densification.
3. Fare feature rows equal clean trip rows; no aggregation occurs.
4. Lag and rolling demand features use prior observations within the same zone.
5. Train time ends no later than validation time; validation ends no later than
   test time.

Interpretation limits: this is one month of Yellow Taxi data, not Green Taxi or
FHV data; zone IDs are coarse geography; no rider/driver demographics or exact
GPS coordinates are present; cleaning bounds encode assumptions; and one month
cannot represent annual seasonality or long-term change.
"""
)
code(
    """
assert len(clean) <= len(raw)
assert len(fare_features) == len(clean)
assert not demand.duplicated(["PULocationID", "pickup_hour"]).any()
for problem in ("demand", "fare"):
    rows = split_evidence[split_evidence["problem"] == problem].set_index("split")
    assert rows.loc["train", "last_time"] <= rows.loc["validation", "first_time"]
    assert rows.loc["validation", "last_time"] <= rows.loc["test", "first_time"]
print("All lineage checks passed.")
"""
)

md(
    """
## Recap

- Official TLC Parquet and CSV files land in `data/raw/` with their original
    structure preserved.
- Validation applies an explicit schema, period, null and plausibility contract.
- The demand branch changes the grain from one trip to one zone-hour, then adds
    past-only calendar, lag and rolling features.
- The fare branch preserves one row per clean trip and predicts base fare.
- Both branches are split chronologically and persisted for model training.

Each finding above is backed by a table, chart or assertion in this notebook.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("Wrote", OUT)
