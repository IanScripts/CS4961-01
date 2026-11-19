"""Create compact California-local daily summaries from saved IEM hourly weather.

This reads the existing normalized, UTC-grid IEM hourly dataset in chunks.
No API calls, interpolation, precipitation summation or wildfire scores.
Only complete California-local calendar-day grids are exported. Note that a
UTC-year-only file lacks the final hours of December 31 California local time.

Usage from CS4961-01 root:
    python scripts/build_iem_statewide_daily.py --year 2025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZONE = "America/Los_Angeles"
COLUMNS = [
    "station_id", "time_utc", "report_count", "source", "data_type",
    "temperature_f", "relative_humidity_pct", "wind_speed_mph",
    "precipitation_report_1h_in",
]
SUM_FIELDS = [
    "hour_slots", "hours_with_reports", "temperature_hours", "temperature_sum_f",
    "humidity_hours", "humidity_sum_pct", "wind_hours", "wind_sum_mph",
    "precipitation_report_hours",
]
AGGREGATES = {
    **{k: "sum" for k in SUM_FIELDS},
    "temperature_max_f": "max",
    "humidity_min_pct": "min",
    "wind_max_mph": "max",
}


def expected_local_hours(date_string: str) -> int:
    date = pd.Timestamp(date_string)
    start = date.tz_localize(ZONE)
    end = (date + pd.Timedelta(days=1)).tz_localize(ZONE)
    return int((end.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1))


def aggregate_chunks(chunks, year: int) -> tuple[pd.DataFrame, int]:
    """Aggregate at station/day granularity without retaining hourly input rows.

    Retains count denominators for field-level missingness. Calendar-local days
    are only included if all 23/24/25 UTC-hour slots exist in the input grid;
    missing *reports* remain missing, not filled.
    """
    parts = []
    read_rows = 0
    for chunk in chunks:
        required = set(COLUMNS) - set(chunk.columns)
        if required:
            raise ValueError("IEM hourly data missing: " + ", ".join(sorted(required)))
        if chunk.empty:
            continue
        read_rows += len(chunk)
        if not chunk["source"].astype(str).eq("IEM ASOS archive").all():
            raise ValueError("Unexpected source: only IEM ASOS archive is supported")
        if not chunk["data_type"].astype(str).eq("observed").all():
            raise ValueError("Unexpected data_type: only observed data is supported")
        time = pd.to_datetime(chunk["time_utc"], utc=True, errors="coerce")
        if time.isna().any() or not time.eq(time.dt.floor("h")).all():
            raise ValueError("Invalid or nonhourly time_utc")
        if chunk["station_id"].isna().any() or chunk["station_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Missing station identifiers")
        d = chunk[["station_id", "report_count", "temperature_f", "relative_humidity_pct", "wind_speed_mph", "precipitation_report_1h_in"]].copy()
        d["local_date"] = time.dt.tz_convert(ZONE).dt.strftime("%Y-%m-%d")
        for col in ["report_count", "temperature_f", "relative_humidity_pct", "wind_speed_mph", "precipitation_report_1h_in"]:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        if d["report_count"].isna().any() or (d["report_count"] < 0).any():
            raise ValueError("Invalid report_count")
        d["has_report"] = d["report_count"].gt(0).astype(int)
        part = d.groupby(["station_id", "local_date"], sort=False).agg(
            hour_slots=("report_count", "size"),
            hours_with_reports=("has_report", "sum"),
            temperature_hours=("temperature_f", "count"),
            temperature_sum_f=("temperature_f", "sum"),
            temperature_max_f=("temperature_f", "max"),
            humidity_hours=("relative_humidity_pct", "count"),
            humidity_sum_pct=("relative_humidity_pct", "sum"),
            humidity_min_pct=("relative_humidity_pct", "min"),
            wind_hours=("wind_speed_mph", "count"),
            wind_sum_mph=("wind_speed_mph", "sum"),
            wind_max_mph=("wind_speed_mph", "max"),
            precipitation_report_hours=("precipitation_report_1h_in", "count"),
        ).reset_index()
        parts.append(part)
    if not parts:
        raise ValueError("Input contained no hourly station records")
    daily = (pd.concat(parts, ignore_index=True)
             .groupby(["station_id", "local_date"], as_index=False)
             .agg(AGGREGATES))
    daily = daily[daily["local_date"].str.startswith(f"{year:04d}-")].copy()
    hours_expected = {day: expected_local_hours(day) for day in daily["local_date"].unique()}
    daily["expected_local_hours"] = daily["local_date"].map(hours_expected)
    partial_count = int((daily["hour_slots"] != daily["expected_local_hours"]).sum())
    daily = daily[daily["hour_slots"] == daily["expected_local_hours"]].copy()
    for name, count, total in [
        ("temperature_mean_f", "temperature_hours", "temperature_sum_f"),
        ("humidity_mean_pct", "humidity_hours", "humidity_sum_pct"),
        ("wind_mean_mph", "wind_hours", "wind_sum_mph"),
    ]:
        daily[name] = (daily[total] / daily[count].replace(0, float("nan"))).round(2)
    daily["report_coverage_pct"] = (100 * daily["hours_with_reports"] / daily["hour_slots"]).round(2)
    daily["data_type"] = "observed"
    daily["source"] = "IEM ASOS archive"
    daily["day_timezone"] = ZONE
    daily = daily.drop(columns=["temperature_sum_f", "humidity_sum_pct", "wind_sum_mph"])
    order = [
        "station_id", "local_date", "day_timezone", "source", "data_type",
        "expected_local_hours", "hour_slots", "hours_with_reports", "report_coverage_pct",
        "temperature_hours", "temperature_mean_f", "temperature_max_f",
        "humidity_hours", "humidity_mean_pct", "humidity_min_pct",
        "wind_hours", "wind_mean_mph", "wind_max_mph", "precipitation_report_hours",
    ]
    return daily[order].sort_values(["station_id", "local_date"]).reset_index(drop=True), partial_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=75000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    folder = ROOT / "live" / "california_sources" / "iem"
    source = args.input or (folder / f"iem_california_hourly_{args.year}.csv")
    output = args.output or (folder / f"iem_daily_station_summary_{args.year}.csv")
    if not source.is_file():
        parser.error(f"IEM hourly input is missing: {source}")
    try:
        summary, partial = aggregate_chunks(
            pd.read_csv(source, usecols=COLUMNS, chunksize=args.chunk_size, low_memory=False), args.year
        )
    except (ValueError, pd.errors.ParserError, OSError) as exc:
        parser.error(str(exc))
    if summary.empty:
        parser.error("No complete California local days found")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Created {len(summary):,} complete station-local-days across {summary.station_id.nunique()} stations")
    print(f"Period: {summary.local_date.min()} to {summary.local_date.max()} (America/Los_Angeles)")
    print(f"Excluded {partial:,} incomplete station-local-days at UTC-year boundaries or incomplete grids")
    print(f"Summary saved: {output}")
    print("Observed weather only; no precipitation totals, interpolated values or wildfire predictions")


if __name__ == "__main__":
    main()
