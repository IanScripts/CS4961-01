"""Create an exploratory multi-cell gridMET/weather + FPA-FOD occurrence table.

Run from the CS4961-01 repository root:
    python -m scripts.build_multicell_weather_pilot

Development fixture only. Not a validated operational wildfire forecast.
"""

import csv
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from src.ai_prediction.build_regional_labels import (
    build_cell_population,
    load_config,
)
from src.ai_prediction.grid import CELL_SIZE
from src.ai_prediction.read_occurrences import iter_occurrences
from src.ai_prediction.regional_labels import (
    aggregate_event_counts,
    iter_label_rows,
)

CONFIG_PATH = Path("config/ai/regions-v1.json")
OUTPUT_DIR = Path("data/ai/weather_pilot_multi")

# Positive-control cell + four exploratory eligible non-LA cells.
# These are QA choices, NOT a representative sampling design.
PILOT_CELLS = [(-184, 136), (-196, 138), (-189, 135), (-179, 128), (-173, 154)]
TARGET_START = date(2018, 7, 1)
TARGET_END = date(2018, 7, 31)
LAG_DAYS = 2
RAINFALL_DAYS = 7
VARIABLES = ["tmmx", "rmin", "vs", "pr"]

# Expected display names and units returned by PyGridMET's point API.
EXPECTED_UNITS = {"tmmx": "K", "rmin": "%", "vs": "m/s", "pr": "mm"}
RANGES = {"tmmx": (180, 350), "rmin": (0, 100), "vs": (0, 80), "pr": (0, 1500)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_center_latlon(cell, transformer):
    ix, iy = cell
    x = (ix + 0.5) * CELL_SIZE
    y = (iy + 0.5) * CELL_SIZE
    longitude, latitude = transformer.transform(x, y)
    return latitude, longitude


def check_weather_frame(frame, start, end):
    """Return {date: {code: value}}; reject unit, time and data errors."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("gridMET returned no daily data")

    columns = {}
    for code in VARIABLES:
        expected = f"{code} ({EXPECTED_UNITS[code]})"
        matches = [c for c in frame.columns if str(c).strip() == expected]
        if len(matches) != 1:
            raise ValueError(
                f"Expected gridMET column {expected!r}; got {list(frame.columns)!r}. "
                "Inspect the provider's variable and unit conventions."
            )
        columns[code] = matches[0]

    try:
        timestamps = pd.to_datetime(frame.index, utc=True, errors="raise")
    except Exception as exc:
        raise ValueError("gridMET index must contain daily timestamps") from exc

    if timestamps.isna().any():
        raise ValueError("Missing gridMET timestamps")

    weather = {}
    for i, stamp in enumerate(timestamps):
        day = stamp.date()
        if day in weather:
            raise ValueError(f"Duplicate gridMET calendar date: {day}")
        entry = {}
        for code, column in columns.items():
            value = float(frame.iloc[i][column])
            lower, upper = RANGES[code]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Invalid {code} on {day}: {value}")
            entry[code] = value
        weather[day] = entry

    current = start
    while current <= end:
        if current not in weather:
            raise ValueError(f"Missing gridMET day: {current}")
        current += timedelta(days=1)
    return weather


def build_joined_rows(labels, weather_by_cell):
    """Align labels with preceding weather days, rejecting missing dates."""
    output = []
    seen_keys = set()
    for label in labels:
        cell = label["grid_cell"]
        target = date.fromisoformat(label["date"])
        key = (cell, target)
        if key in seen_keys:
            raise ValueError(f"Duplicate target cell/day: {key}")
        seen_keys.add(key)

        feature_day = target - timedelta(days=LAG_DAYS)
        series = weather_by_cell[cell]
        if feature_day not in series:
            raise ValueError(f"Missing weather: {cell} / {feature_day}")
        current = series[feature_day]

        rainfall = []
        for offset in range(RAINFALL_DAYS):
            day = feature_day - timedelta(days=offset)
            if day not in series:
                raise ValueError(f"Missing rainfall history: {cell} / {day}")
            rainfall.append(series[day]["pr"])

        count = int(label["fire_count"])
        fire_label = int(label["fire_label"])
        if count < 0 or fire_label != int(count > 0):
            raise ValueError(f"Inconsistent fire label: {label}")

        output.append({
            "grid_cell": cell,
            "target_date_local": target.isoformat(),
            "weather_date_gridmet": feature_day.isoformat(),
            "tmax_k": round(current["tmmx"], 3),
            "rh_min_pct": round(current["rmin"], 3),
            "wind_mean_ms": round(current["vs"], 3),
            "precip_mm": round(current["pr"], 3),
            "precip_previous_7d_mm": round(sum(rainfall), 3),
            "fire_count": count,
            "fire_label": fire_label,
            "weather_source": "gridMET",
            "evidence_class": "retrospective_reconstruction",
            "spatial_method": "single_gridmet_pixel_at_10km_cell_center",
            "weather_lag_days": LAG_DAYS,
        })
    return output


def main():
    # Import here so the transformation and join functions can be tested offline.
    import pygridmet as gridmet

    config = load_config(CONFIG_PATH)
    regions, eligible, _ = build_cell_population(config)
    for cell in PILOT_CELLS:
        if cell not in eligible:
            raise ValueError(f"Pilot cell is excluded by geographic rules: {cell}")

    # Weather must cover all seven days preceding the feature date.
    weather_start = TARGET_START - timedelta(days=LAG_DAYS + RAINFALL_DAYS - 1)
    weather_end = TARGET_END - timedelta(days=LAG_DAYS)
    converter = Transformer.from_crs("EPSG:5070", "EPSG:4269", always_xy=True)

    weather_by_cell = {}
    for cell in PILOT_CELLS:
        lat, lon = cell_center_latlon(cell, converter)
        cell_id = f"{cell[0]}_{cell[1]}"
        print(f"Fetching gridMET {cell_id} ({lat:.4f}, {lon:.4f})")
        frame = gridmet.get_bycoords(
            (lon, lat),
            (weather_start.isoformat(), weather_end.isoformat()),
            variables=VARIABLES,
            validate_filesize=True,
        )
        weather_by_cell[cell_id] = check_weather_frame(frame, weather_start, weather_end)

    # Read the nationwide date window once; geographic assignment controls inclusion.
    events = iter_occurrences(config["sources"]["wildfires"], TARGET_START, TARGET_END)
    counts = aggregate_event_counts(events, regions, eligible)
    labels = iter_label_rows(
        {"non_la": PILOT_CELLS}, TARGET_START, TARGET_END, counts
    )
    joined = build_joined_rows(labels, weather_by_cell)
    expected = len(PILOT_CELLS) * (TARGET_END - TARGET_START).days + len(PILOT_CELLS)
    if len(joined) != expected:
        raise ValueError(f"Expected {expected} cell-days, got {len(joined)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "training_features_pilot.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    manifest = {
        "purpose": "Exploratory weather/label join, NOT validated wildfire forecast",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "weather_product": "gridMET", "pygridmet_version": version("pygridmet"),
        "variables": VARIABLES, "weather_units": EXPECTED_UNITS,
        "weather_start": weather_start.isoformat(),
        "weather_end": weather_end.isoformat(),
        "target_start": TARGET_START.isoformat(),
        "target_end": TARGET_END.isoformat(),
        "weather_lag_days": LAG_DAYS,
        "gridmet_day_convention": "nominal midnight-to-midnight MST (ends 07Z following day)",
        "spatial_method": "single_gridmet_pixel_at_10km_cell_center",
        "sampling_note": "Positive-control and exploratory cells; not unbiased training sample",
        "sample_cells": [f"{ix}_{iy}" for ix, iy in PILOT_CELLS],
        "features_csv_sha256": file_sha256(output),
        "rows": len(joined),
        "positive_cell_days": sum(r["fire_label"] for r in joined),
        "negative_cell_days": sum(not r["fire_label"] for r in joined),
        "config_sha256": file_sha256(CONFIG_PATH),
        "fpa_fod_sha256": file_sha256(config["sources"]["wildfires"]),
        "census_counties_sha256": file_sha256(config["sources"]["counties"]),
        "limitation": "Historical gridMET values are not proven available at the stated issue time",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(joined)} joined cell-days to {output}")
    print(f"Positive: {manifest['positive_cell_days']}; Negative: {manifest['negative_cell_days']}")
    print("Manifest:", OUTPUT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
