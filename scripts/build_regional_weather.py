"""Build monthly area-weighted gridMET + FPA-FOD cell-day CSVs.

From the repository root:
    python -m scripts.build_regional_weather --month 2018-07 --max-cells 5
    python -m scripts.build_regional_weather --month 2018-08
    python -m scripts.build_regional_weather --month 2019-01

One command does geographic tiling, cached downloads, spatial aggregation,
fire-label joins, completeness checks, and a reproducibility manifest.
This is retrospective research data, not an as-issued weather forecast.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from calendar import monthrange
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from pyproj import Transformer

from src.ai_prediction.build_regional_labels import build_cell_population, load_config
from src.ai_prediction.grid import CELL_SIZE
from src.ai_prediction.gridmet_weights import load_or_build_weights
from src.ai_prediction.read_occurrences import iter_occurrences
from src.ai_prediction.regional_labels import aggregate_event_counts

import time
from requests.exceptions import RequestException

CONFIG = Path("config/ai/regions-v1.json")
OUT_ROOT = Path("data/ai")
LAG = 2
RAIN_DAYS = 7
MIN_COVERAGE = 0.95
# Each request covers a one-degree tile plus 0.20 degree edge padding.
# Padding ensures cells near tile edges have all their intersecting pixels.
PAD = 0.20
SERVICE = "https://www.reacchpna.org/thredds/ncss/grid/MET"
VARIABLES = {
    "tmmx": ("tmax_k", "K", 180.0, 350.0),
    "rmin": ("rh_min_pct", "%", 0.0, 100.0),
    "vs": ("wind_mean_ms", "m/s", 0.0, 80.0),
    "pr": ("precip_mm", "mm", 0.0, 1500.0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_is_netcdf(path: Path) -> bool:
    with path.open("rb") as stream:
        head = stream.read(8)
    return head.startswith(b"CDF") or head.startswith(b"\x89HDF")


def tiles_for_cells(cells):
    """Assign each cell to a deterministic 1-degree lon/lat tile."""
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    tiles = defaultdict(list)
    for ix, iy in cells:
        x = (ix + .5) * CELL_SIZE
        y = (iy + .5) * CELL_SIZE
        lon, lat = transformer.transform(x, y)
        tiles[(math.floor(lon), math.floor(lat))].append((ix, iy))
    return {tile: sorted(group) for tile, group in sorted(tiles.items())}


def download_subset(code, year, tile, weather_start, weather_end, download_dir):
    lon0, lat0 = tile
    url = f"{SERVICE}/{code}/{code}_{year}.nc"
    # For these specific 2018 gridMET variables, the THREDDS NCSS variable
    # is discovered from the provider metadata instead of being guessed.
    name = {"tmmx": "air_temperature", "rmin": "relative_humidity",
            "vs": "wind_speed", "pr": "precipitation_amount"}[code]
    params = {
        "var": name, "north": lat0 + 1 + PAD, "south": lat0 - PAD,
        "east": lon0 + 1 + PAD, "west": lon0 - PAD,
        "time_start": weather_start.isoformat() + "T00:00:00Z",
        "time_end": weather_end.isoformat() + "T00:00:00Z",
        "accept": "netcdf", "addLatLon": "true",
    }
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:16]
    path = download_dir / f"{code}_{year}_{lon0}_{lat0}_{key}.nc"
    if path.exists():
        if not file_is_netcdf(path):
            raise ValueError(f"Cached response is not NetCDF: {path}")
        return path, False



    print(f"  Downloading {code} tile {tile} ...", flush=True)

    max_attempts = 5
    retry_statuses = {429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=180
            )

            if response.ok:
                data = response.content
                break

            # Don't retry errors such as 400 or 404.
            if response.status_code not in retry_statuses:
                raise RuntimeError(
                    f"gridMET {code} {tile}: "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            error = f"HTTP {response.status_code}"

            # Respect Retry-After when supplied by the server.
            retry_after = response.headers.get("Retry-After")

        except RequestException as exc:
            error = str(exc)
            retry_after = None

        if attempt == max_attempts:
            raise RuntimeError(
                f"gridMET {code} {tile} failed "
                f"after {max_attempts} attempts: {error}"
            )

        delay = min(5 * (2 ** (attempt - 1)), 60)

        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), 180))
            except ValueError:
                pass

        print(
            f"  Temporary gridMET failure: {error}. "
            f"Retrying in {delay:.0f}s "
            f"({attempt}/{max_attempts - 1})..."
        )

        time.sleep(delay)

    
    if not (data.startswith(b"CDF") or data.startswith(b"\x89HDF")):
        raise ValueError(f"gridMET returned non-NetCDF data for {code} {tile}: {data[:180]!r}")
    tmp = path.with_suffix(".partial")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path, True


def read_aggregate(path, code, cells, weights_dir):
    column, expected_units, low, high = VARIABLES[code]
    with xr.open_dataset(path, decode_cf=True) as ds:
        possible = [name for name, data in ds.data_vars.items()
                    if set(("day", "lat", "lon")).issubset(data.dims)]
        if len(possible) != 1:
            raise ValueError(f"Unexpected {code} raster variables: {possible}")
        var = ds[possible[0]].transpose("day", "lat", "lon")
        units = str(var.attrs.get("units", "")).strip()
        if units != expected_units:
            raise ValueError(f"Unexpected {code} units {units!r}, expected {expected_units!r}")
        dates = [date.fromisoformat(np.datetime_as_string(t, unit="D")[:10])
                 for t in ds["day"].values]
        if len(dates) != len(set(dates)):
            raise ValueError(f"Duplicate weather dates for {code}")
        if not dates:
            raise ValueError(f"Empty weather raster for {code}")
        values = np.asarray(var.values, dtype=np.float64)
        valid = np.isfinite(values)
        if np.any(valid & ((values < low) | (values > high))):
            raise ValueError(f"{code} contains decoded values outside expected {expected_units} range")
        weights, reused, cache_path = load_or_build_weights(
            ds["lat"].values, ds["lon"].values, cells, weights_dir)
        means, coverage, counts = weights.aggregate(values, min_coverage=MIN_COVERAGE)
        if np.any(coverage > 1.000001):
            raise ValueError("Raster pixel overlaps exceed full-cell coverage")
        result = {}
        for ti, day in enumerate(dates):
            for ci, cell in enumerate(cells):
                result[(cell, day)] = (float(means[ti, ci]), float(coverage[ti, ci]))
        return result, reused, str(cache_path), column


def build_dataset(month="2018-07", max_cells=None):
    try:
        parsed_month = date.fromisoformat(f"{month}-01")
    except ValueError as exc:
        raise ValueError("--month must be YYYY-MM, such as 2018-08") from exc
    if parsed_month.strftime("%Y-%m") != month:
        raise ValueError("--month must be YYYY-MM, such as 2018-08")
    start = parsed_month
    end = date(start.year, start.month, monthrange(start.year, start.month)[1])
    out_dir = OUT_ROOT / f"regional_weather_{start.year}_{start.month:02d}"
    config = load_config(CONFIG)
    regions, eligible, _ = build_cell_population(config)
    cells = sorted(eligible)
    if max_cells is not None:
        if max_cells < 1:
            raise ValueError("--max-cells must be positive")
        # Always include our real-fire positive-control cell for a smoke test.
        if (-184, 136) not in eligible:
            raise ValueError("Riverside positive-control cell excluded")
        cells = sorted({(-184, 136), *[cell for cell in cells if cell != (-184, 136)][:max_cells - 1]})
    tiles = tiles_for_cells(cells)
    weather_start = start - timedelta(days=LAG + RAIN_DAYS - 1)
    weather_end = end - timedelta(days=LAG)
    download_dir = out_dir / "raw_subsets"
    weights_dir = OUT_ROOT / "gridmet_weight_cache"
    for directory in (out_dir, download_dir, weights_dir):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Cells: {len(cells)} | tiles: {len(tiles)} | target: {start}..{end}")
    print(f"Weather dates: {weather_start}..{weather_end} (lag {LAG} days)")

    # Four dictionaries keyed (cell tuple, weather date). Each tile is processed
    # one variable at a time; we do not hold regional 3D grids in memory.
    weather = {code: {} for code in VARIABLES}
    raw_files = []
    cache_summary = Counter()
    for tile, tile_cells in tiles.items():
        print(f"Tile {tile}: {len(tile_cells)} target cells", flush=True)
        for code in VARIABLES:
            # A January target may require December weather from the prior year.
            # gridMET publishes a separate file for each calendar year.
            for year in range(weather_start.year, weather_end.year + 1):
                request_start = max(weather_start, date(year, 1, 1))
                request_end = min(weather_end, date(year, 12, 31))
                if request_start > request_end:
                    continue
                path, downloaded = download_subset(
                    code, year, tile, request_start, request_end, download_dir)
                aggregates, reused, cache_path, _ = read_aggregate(
                    path, code, tile_cells, weights_dir)
                cache_summary["reused" if reused else "created"] += 1
                duplicate = aggregates.keys() & weather[code].keys()
                if duplicate:
                    raise ValueError(f"Duplicate weather cell-days for {code}: {len(duplicate)}")
                weather[code].update(aggregates)
                raw_files.append({"variable": code, "year": year, "tile": list(tile),
                                  "path": str(path), "sha256": sha256(path),
                                  "downloaded_now": downloaded, "weight_cache": cache_path})

    # Read wildfire events once across the target-date window.
    events = iter_occurrences(config["sources"]["wildfires"], start, end)
    fire_counts = aggregate_event_counts(events, regions, eligible)
    print("Joining weather features to FPA-FOD fire labels ...")
    output = out_dir / ("training_features_smoke.csv" if max_cells else "training_features.csv")
    columns = ["grid_cell", "target_date_local", "weather_date_gridmet",
               "tmax_k", "rh_min_pct", "wind_mean_ms", "precip_mm",
               "precip_previous_7d_mm", "fire_count", "fire_label",
               "coverage_min_fraction", "weather_lag_days", "spatial_method"]
    expected = len(cells) * ((end - start).days + 1)
    n_rows = n_positive = n_missing = 0
    # We keep all cell-days, with blank features when weather coverage is
    # inadequate, instead of dropping them (which could bias labels).
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for target_offset in range((end - start).days + 1):
            target = start + timedelta(days=target_offset)
            feature_day = target - timedelta(days=LAG)
            for cell in cells:
                label_count = int(fire_counts.get(("non_la", f"{cell[0]}_{cell[1]}", target), 0))
                if label_count < 0:
                    raise ValueError("Negative FPA-FOD event count")
                row = {"grid_cell": f"{cell[0]}_{cell[1]}",
                       "target_date_local": target.isoformat(),
                       "weather_date_gridmet": feature_day.isoformat(),
                       "fire_count": label_count, "fire_label": int(label_count > 0),
                       "weather_lag_days": LAG,
                       "spatial_method": "area_weighted_gridmet_pixels"}
                coverages = []
                for code, (field, _, _, _) in VARIABLES.items():
                    key = (cell, feature_day)
                    if key not in weather[code]:
                        raise ValueError(f"Missing gridMET date: {code} {cell} {feature_day}")
                    value, coverage = weather[code][key]
                    coverages.append(coverage)
                    row[field] = "" if not math.isfinite(value) else round(value, 3)
                rainfall = []
                for lag in range(RAIN_DAYS):
                    key = (cell, feature_day - timedelta(days=lag))
                    if key not in weather["pr"]:
                        raise ValueError(f"Missing rainfall date: {cell} {key[1]}")
                    amount, coverage = weather["pr"][key]
                    rainfall.append(amount)
                    coverages.append(coverage)
                row["precip_previous_7d_mm"] = (
                    "" if not all(math.isfinite(v) for v in rainfall)
                    else round(sum(rainfall), 3))
                row["coverage_min_fraction"] = round(min(coverages), 6)
                if any(row[field] == "" for field in
                       ("tmax_k", "rh_min_pct", "wind_mean_ms", "precip_mm", "precip_previous_7d_mm")):
                    n_missing += 1
                writer.writerow(row)
                n_rows += 1
                n_positive += row["fire_label"]
    if n_rows != expected:
        raise ValueError(f"Expected {expected} cell-days; wrote {n_rows}")
    manifest = {
        "dataset": f"regional_gridmet_fire_labels_{month}",
        "purpose": "Retrospective research; NOT an as-issued operational forecast",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(CONFIG), "config_sha256": sha256(CONFIG),
        "fire_data_sha256": sha256(Path(config["sources"]["wildfires"])),
        "county_data_sha256": sha256(Path(config["sources"]["counties"])),
        "start": start.isoformat(), "end": end.isoformat(),
        "weather_start": weather_start.isoformat(), "weather_end": weather_end.isoformat(),
        "gridmet_day_convention": "nominal midnight-to-midnight MST",
        "weather_lag_days": LAG, "rainfall_window_days": RAIN_DAYS,
        "min_valid_area_coverage": MIN_COVERAGE,
        "variables": {c: {"field": v[0], "units": v[1]} for c, v in VARIABLES.items()},
        "spatial_method": "area_weighted_gridmet_pixels",
        "spatial_caveat": "GridMET pixel edges inferred from coordinates; not independently validated against native cell bounds",
        "sampling": "entire eligible non-LA population" if max_cells is None else "smoke sample including known positive cell (not representative)",
        "cells": len(cells), "tiles": len(tiles), "rows": n_rows,
        "positive_cell_days": n_positive, "negative_cell_days": n_rows - n_positive,
        "rows_with_incomplete_weather": n_missing,
        "weight_caches": dict(cache_summary), "raw_subsets": raw_files,
        "output": str(output), "output_sha256": sha256(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rows: {n_rows:,} | positive: {n_positive:,} | negative: {n_rows - n_positive:,}")
    print(f"Incomplete weather rows: {n_missing:,} | weight cache: {dict(cache_summary)}")
    print("CSV:", output)
    print("Manifest:", manifest_path)


def month_sequence(start_month, end_month):
    """Inclusive list of YYYY-MM months."""
    for value in (start_month, end_month):
        try:
            parsed = date.fromisoformat(value + "-01")
        except ValueError as exc:
            raise ValueError(f"Invalid month: {value!r}; expected YYYY-MM") from exc
        if parsed.strftime("%Y-%m") != value:
            raise ValueError(f"Invalid month: {value!r}; expected YYYY-MM")
    start_year, start_num = map(int, start_month.split("-"))
    end_year, end_num = map(int, end_month.split("-"))
    if (start_year, start_num) > (end_year, end_num):
        raise ValueError("--from-month must not be after --through-month")
    months = []
    year, number = start_year, start_num
    while (year, number) <= (end_year, end_num):
        months.append(f"{year:04d}-{number:02d}")
        number += 1
        if number == 13:
            year, number = year + 1, 1
    return months


def monthly_paths(month, smoke):
    year, number = month.split("-")
    folder = OUT_ROOT / f"regional_weather_{year}_{number}"
    name = "training_features_smoke.csv" if smoke else "training_features.csv"
    csv_path = folder / name
    return csv_path, csv_path.with_suffix(".manifest.json")


def inspect_completed_month(month, smoke, *, check_hash=True):
    """Return manifest when both complete artifacts agree; else None.

    Does not infer success from file existence alone. This is a resume guard,
    not a substitute for a scientific QA of the weather and labels.
    """
    csv_path, manifest_path = monthly_paths(month, smoke)
    if not csv_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        year, number = map(int, month.split("-"))
        days = monthrange(year, number)[1]
        if manifest.get("dataset") != f"regional_gridmet_fire_labels_{month}":
            return None
        if not isinstance(manifest.get("cells"), int) or manifest["cells"] <= 0:
            return None
        if manifest.get("rows") != manifest["cells"] * days:
            return None
        if manifest.get("positive_cell_days", -1) + manifest.get("negative_cell_days", -1) != manifest["rows"]:
            return None
        expected_sample = "entire eligible non-LA population" if not smoke else None
        if not smoke and manifest.get("sampling") != expected_sample:
            return None
        if smoke and not str(manifest.get("sampling", "")).startswith("smoke sample"):
            return None
        if manifest.get("config_sha256") != sha256(CONFIG):
            return None
        config = load_config(CONFIG)
        if (manifest.get("fire_data_sha256") != sha256(Path(config["sources"]["wildfires"]))
                or manifest.get("county_data_sha256") != sha256(Path(config["sources"]["counties"]))):
            return None
        if (manifest.get("weather_lag_days") != LAG
                or manifest.get("rainfall_window_days") != RAIN_DAYS
                or manifest.get("min_valid_area_coverage") != MIN_COVERAGE):
            return None
        if not csv_path.stat().st_size or not manifest.get("output_sha256"):
            return None
        if check_hash and manifest["output_sha256"] != sha256(csv_path):
            return None
        # Verify header against the expected builder output schema.
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            columns = next(csv.reader(f))
        if columns != ["grid_cell", "target_date_local", "weather_date_gridmet",
                       "tmax_k", "rh_min_pct", "wind_mean_ms", "precip_mm",
                       "precip_previous_7d_mm", "fire_count", "fire_label",
                       "coverage_min_fraction", "weather_lag_days", "spatial_method"]:
            return None
        return manifest
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, StopIteration):
        return None


def build_batch(first, last, max_cells=None, force=False):
    months = month_sequence(first, last)
    report_dir = OUT_ROOT / "regional_weather_batch"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (f"report_{first}_to_{last}" +
                                ("_smoke" if max_cells is not None else "") + ".csv")
    report = []
    print(f"Batch: {len(months)} months ({first} .. {last})")
    for month in months:
        existing = None if force else inspect_completed_month(month, max_cells is not None)
        if existing is not None and (max_cells is None or existing["cells"] == max_cells):
            print(f"[{month}] verified existing CSV and manifest; skipping")
            manifest = existing
            status = "reused"
        else:
            print(f"[{month}] building")
            build_dataset(month, max_cells=max_cells)
            manifest = inspect_completed_month(month, max_cells is not None)
            if manifest is None or (max_cells is not None and manifest["cells"] != max_cells):
                raise RuntimeError(f"[{month}] artifact verification failed after build")
            status = "built"
        report.append({"month": month, "status": status, "cells": manifest["cells"],
                       "rows": manifest["rows"], "positive_cell_days": manifest["positive_cell_days"],
                       "negative_cell_days": manifest["negative_cell_days"],
                       "incomplete_weather_rows": manifest["rows_with_incomplete_weather"],
                       "csv_sha256": manifest["output_sha256"]})
        # Keep a useful report even if a later month's download fails.
        with report_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(report)
    print(f"Batch report: {report_path}")
    print(f"Months: {len(report)} | rows: {sum(r['rows'] for r in report):,} | "
          f"positive cell-days: {sum(r['positive_cell_days'] for r in report):,} | "
          f"incomplete: {sum(r['incomplete_weather_rows'] for r in report):,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Smoke test only: number of cells, including Riverside positive control")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--month", default=None, help="One target month YYYY-MM")
    group.add_argument("--from-month", default=None, help="First month in batch YYYY-MM")
    parser.add_argument("--through-month", default=None, help="Last month in batch YYYY-MM")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild outputs even if verified CSV and manifest exist")
    args = parser.parse_args()
    if args.max_cells is not None and args.max_cells <= 0:
        parser.error("--max-cells must be positive")
    if args.from_month:
        if not args.through_month:
            parser.error("--from-month requires --through-month")
        build_batch(args.from_month, args.through_month, args.max_cells, args.force)
    else:
        if args.through_month:
            parser.error("--through-month requires --from-month")
        month = args.month or "2018-07"
        month_sequence(month, month)  # validate before downloading
        if not args.force and inspect_completed_month(month, args.max_cells is not None):
            manifest = inspect_completed_month(month, args.max_cells is not None)
            if args.max_cells is None or manifest["cells"] == args.max_cells:
                print(f"[{month}] verified existing CSV and manifest; skipping (use --force to rebuild)")
                return
        build_dataset(month, args.max_cells)


if __name__ == "__main__":
    main()
