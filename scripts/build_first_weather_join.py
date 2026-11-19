
import csv
import io
from datetime import date, timedelta
from pathlib import Path

import requests
from pyproj import Transformer

from src.ai_prediction.grid import CELL_SIZE
from src.ai_prediction.build_regional_labels import (
    load_config,
    build_cell_population,
    generate_pilot_dataset
)


CONFIG = "config/ai/regions-v1.json"
OUTPUT = Path("data/ai/weather_pilot")

# Verified Riverside wildfire cell
PILOT_CELL = (-184, 136)

START_DATE = date(2018, 7, 1)
END_DATE = date(2018, 7, 4)

# gridMET 2018 maximum temperature
GRIDMET_URL = (
    "https://www.reacchpna.org/thredds/ncss/"
    "MET/tmmx/tmmx_2018.nc"
)


def get_cell_center(ix, iy):
    # Find the center of our 10 km grid cell
    x = (ix + 0.5) * CELL_SIZE
    y = (iy + 0.5) * CELL_SIZE

    transformer = Transformer.from_crs(
        "EPSG:5070",
        "EPSG:4269",
        always_xy=True
    )

    longitude, latitude = transformer.transform(x, y)

    return latitude, longitude


def download_gridmet(latitude, longitude):
    # Read a few days before our target dates
    params = {
        "var": "air_temperature",
        "latitude": latitude,
        "longitude": longitude,
        "time_start": "2018-06-28T00:00:00Z",
        "time_end": "2018-07-04T00:00:00Z",
        "accept": "csv"
    }

    print("Requesting gridMET historical temperature...")

    response = requests.get(
        GRIDMET_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    # The service should return CSV, not an HTML error page
    reader = csv.DictReader(io.StringIO(response.text))
    columns = reader.fieldnames or []

    time_column = next(
        (c for c in columns if c.lower().startswith("time")),
        None
    )

    temperature_column = next(
        (c for c in columns if c.lower().startswith("air_temperature")),
        None
    )

    if not time_column or not temperature_column:
        raise ValueError(
            f"Unexpected gridMET columns: {columns}. "
            f"Response preview: {response.text[:300]}"
        )

    weather = {}

    for row in reader:
        weather_date = date.fromisoformat(
            row[time_column][:10]
        )


        # The 2018 gridMET file stores temperature as
        # packed values with a scale factor and offset.
        raw_temperature = float(row[temperature_column])

        if raw_temperature == 32767:
            raise ValueError(f"Missing gridMET temperature on {weather_date}")

        temperature = raw_temperature * 0.1 + 220.0

        # Check the decoded temperature in Kelvin
        if not 180 <= temperature <= 350:
            raise ValueError(
                f"Invalid decoded temperature on {weather_date}: "
                f"{temperature}"
            )


        # Reject missing values and implausible temperatures
        if not 180 <= temperature <= 350:
            raise ValueError(
                f"Invalid temperature on {weather_date}: "
                f"{temperature}"
            )

        if weather_date in weather:
            raise ValueError(
                f"Duplicate gridMET date: {weather_date}"
            )

        weather[weather_date] = temperature

    if not weather:
        raise ValueError("No gridMET weather records returned")

    print("Weather records:", len(weather))
    print("Temperature units: Kelvin")

    return weather


def main():
    config = load_config(CONFIG)

    regions, eligible_cells, _ = build_cell_population(config)

    if PILOT_CELL not in eligible_cells:
        raise ValueError("Pilot cell is not eligible")

    latitude, longitude = get_cell_center(*PILOT_CELL)

    print("\nRiverside Weather + Wildfire Pilot")
    print("-" * 45)
    print("Cell:", f"{PILOT_CELL[0]}_{PILOT_CELL[1]}")
    print("Latitude:", round(latitude, 5))
    print("Longitude:", round(longitude, 5))

    # Download real historical temperature
    weather = download_gridmet(latitude, longitude)

    # Generate wildfire labels using our existing pipeline
    pilot_config = {
        **config,
        "pilot": {
            **config["pilot"],
            "windows": [
                {
                    "name": "riverside_pilot",
                    "start_date": START_DATE.isoformat(),
                    "end_date": END_DATE.isoformat()
                }
            ]
        }
    }

    generate_pilot_dataset(
        pilot_config,
        regions,
        eligible_cells,
        [PILOT_CELL],
        output_dir=OUTPUT
    )

    # Read the generated fire labels
    labels_path = OUTPUT / "fire_labels.csv"

    with labels_path.open(
        newline="", encoding="utf-8"
    ) as file:
        labels = list(csv.DictReader(file))

    # Join weather and wildfire records
    joined_rows = []

    for label in labels:
        target_date = date.fromisoformat(label["date"])

        # Use an earlier complete weather day.
        # This is still reconstructed history, not an
        # operational as-issued forecast.
        weather_date = target_date - timedelta(days=2)

        if weather_date not in weather:
            raise ValueError(
                f"Missing weather on {weather_date}"
            )

        joined_rows.append({
            "grid_cell": label["grid_cell"],
            "target_date": label["date"],
            "weather_date": weather_date.isoformat(),
            "tmax_kelvin": weather[weather_date],
            "fire_count": label["fire_count"],
            "fire_label": label["fire_label"],
            "weather_source": "gridMET_tmmx_2018",
            "evidence_class": "retrospective_reanalysis",
            "spatial_method": "point_at_cell_center"
        })

    # Save our first combined AI dataset
    output_path = OUTPUT / "training_features_pilot.csv"

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(joined_rows[0])
        )

        writer.writeheader()
        writer.writerows(joined_rows)

    print("\nJoined Weather + Fire Dataset")
    print("-" * 45)

    for row in joined_rows:
        print(
            row["target_date"],
            "Temperature:", row["tmax_kelvin"], "K",
            "Fire label:", row["fire_label"]
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
