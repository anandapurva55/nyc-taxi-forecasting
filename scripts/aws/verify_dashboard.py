"""Exercise the dashboard's native table-rendering path without a live API."""

from importlib.metadata import version
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
from streamlit.testing.v1 import AppTest

from dashboard import data


def main() -> None:
    for package in ("numpy", "pandas", "pyarrow", "streamlit"):
        print(f"{package}=={version(package)}", flush=True)

    zones = data.zone_lookup()["LocationID"].astype(int).tolist()
    response = {
        "request_id": "dashboard-smoke-test",
        "model_reference": "synthetic-test",
        "points": [
            {
                "zone": zone,
                "pickup_hour": "2024-02-01T00:00:00",
                "predicted_pickups": float(zone % 13),
            }
            for zone in zones
        ],
    }
    frame = data.enrich_with_zones(pd.DataFrame(response["points"]))
    top = frame.sort_values("predicted_pickups", ascending=False).head(15)
    columns = ["zone", "Zone", "Borough", "predicted_pickups"]
    for attempt in range(100):
        table = pa.Table.from_pandas(top[columns])
        assert table.num_rows == 15, f"Unexpected row count on attempt {attempt}"

    with patch("dashboard.api_client.forecast_demand", return_value=response):
        app = AppTest.from_string("from dashboard.sections import monitoring\nmonitoring()")
        app.run(timeout=60)
        assert not app.exception, [item.message for item in app.exception]
        assert len(app.dataframe) >= 1, "Monitoring table was not rendered"
        assert len(app.dataframe[0].value) == 15, "Expected 15 top zones"

    print("PASS: 100 Arrow conversions and Monitoring page rendering", flush=True)


if __name__ == "__main__":
    main()
