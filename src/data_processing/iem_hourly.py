"""Normalize saved IEM CA_ASOS reports into quality-flagged station-hour data.

The input is the existing src/weather_apis/iem_california.py observation CSV.
No downloads, NFDRS calculations or experimental fire scores are performed.

From the project root:
    python src/data_processing/iem_hourly.py --stations AAT BUR \\
        --start-date 2026-10-01 --end-date 2026-11-07

Dates are inclusive *UTC* dates. Each station gets one row per UTC hour,
including explicit missing-hour rows (measurements remain empty/NaN).
Instantaneous fields use the last available non-missing measurement in an hour.
IEM's p01i is a one-hour precipitation report: the last available p01i is
retained, NOT the sum of reports within an hour. Do not infer a missing p01i
is zero. This is an exploratory aggregation, not a validated hourly climate
record; see docs for limitations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "live" / "california_sources" / "iem" / "iem_california_observations.csv"
DEFAULT_OUTPUT = ROOT / "live" / "california_sources" / "iem" / "iem_california_hourly.csv"
LOCAL_TZ = "America/Los_Angeles"
KNOTS_TO_MPH = 1.150779448

# Output column, input IEM column
FIELDS = {
    "temperature_f": "tmpf",
    "relative_humidity_pct": "relh",
    "wind_speed_knots": "sknt",
    "wind_direction_deg": "drct",
    "precipitation_report_1h_in": "p01i",
    "gust_speed_knots": "gust",
}


def _last_valid(series: pd.Series) -> float:
    good = series.dropna()
    return float(good.iloc[-1]) if not good.empty else np.nan


def _validate_optional_numeric(frame: pd.DataFrame) -> None:
    """Reject impossible numeric ranges; leave absent values missing."""
    bounds = {
        "relh": (0, 100),
        "sknt": (0, None),
        "gust": (0, None),
        "drct": (0, 360),
        "p01i": (0, None),
    }
    for name, (lower, upper) in bounds.items():
        if name not in frame.columns:
            continue
        values = frame[name].dropna()
        if (values < lower).any() or (upper is not None and (values > upper).any()):
            raise ValueError(f"IEM field {name!r} has physically invalid values; review source CSV")


def normalize_iem_hourly(
    reports: pd.DataFrame,
    stations: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build a per-station UTC hourly grid, retaining missingness and provenance.

    The start/end bounds are inclusive UTC calendar dates. Without bounds,
    use earliest/latest UTC dates among the selected input reports.
    """
    mandatory = {"station_id", "valid"}
    missing = mandatory.difference(reports.columns)
    if missing:
        raise ValueError("Missing IEM input columns: " + ", ".join(sorted(missing)))
    if bool(start_date) != bool(end_date):
        raise ValueError("Supply both --start-date and --end-date or neither")

    d = reports.copy()
    d["station_id"] = d["station_id"].astype("string").str.strip()
    d["valid_utc"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    d = d.dropna(subset=["station_id", "valid_utc"])
    d = d[d["station_id"] != ""]
    if stations:
        wanted = list(dict.fromkeys(str(s).strip() for s in stations))
        unavailable = set(wanted) - set(d["station_id"])
        if unavailable:
            raise ValueError("No input observations for station(s): " + ", ".join(sorted(unavailable)))
        d = d[d["station_id"].isin(wanted)]
    else:
        wanted = sorted(d["station_id"].unique().tolist())
    if d.empty:
        raise ValueError("No valid IEM observations remain after filtering")

    if start_date is None:
        start_day = d["valid_utc"].min().floor("D")
        end_day = d["valid_utc"].max().floor("D")
    else:
        start_day = pd.Timestamp(start_date, tz="UTC")
        end_day = pd.Timestamp(end_date, tz="UTC")
        if start_day != start_day.floor("D") or end_day != end_day.floor("D"):
            raise ValueError("Use YYYY-MM-DD dates (without times)")
    if end_day < start_day:
        raise ValueError("End date must be on or after start date")
    last_exclusive = end_day + pd.Timedelta(days=1)
    d = d[(d["valid_utc"] >= start_day) & (d["valid_utc"] < last_exclusive)].copy()

    for input_name in FIELDS.values():
        if input_name not in d:
            d[input_name] = np.nan
        d[input_name] = pd.to_numeric(d[input_name], errors="coerce")
    _validate_optional_numeric(d)

    # Stable order also makes ties deterministic when input includes same-time reports.
    d = d.sort_values(["station_id", "valid_utc"], kind="stable")
    d["hour_utc"] = d["valid_utc"].dt.floor("h")

    records: list[dict] = []
    for (station, hour), group in d.groupby(["station_id", "hour_utc"], sort=False):
        record = {
            "station_id": station,
            "hour_utc": hour,
            "report_count": int(len(group)),
            "last_report_utc": group["valid_utc"].iloc[-1].isoformat(),
        }
        if "station_name" in group.columns:
            names = group["station_name"].dropna()
            record["station_name"] = str(names.iloc[-1]) if not names.empty else station
        else:
            record["station_name"] = station
        for out_col, source_col in FIELDS.items():
            record[out_col] = _last_valid(group[source_col])
        records.append(record)

    measured = pd.DataFrame.from_records(records)
    hourly = pd.date_range(start_day, last_exclusive, freq="h", inclusive="left", tz="UTC")
    all_slots = pd.MultiIndex.from_product([wanted, hourly], names=["station_id", "hour_utc"])
    result = (
        measured.set_index(["station_id", "hour_utc"])
        .reindex(all_slots)
        .reset_index()
        .sort_values(["station_id", "hour_utc"])
        .reset_index(drop=True)
    )
    name_map = measured.drop_duplicates("station_id").set_index("station_id")["station_name"].to_dict()
    result["station_name"] = result["station_id"].map(name_map)
    result["report_count"] = result["report_count"].fillna(0).astype(int)
    result["missing_hour"] = result["report_count"] == 0
    for col in FIELDS:
        result["missing_" + col] = result[col].isna()
    result["wind_speed_mph"] = result["wind_speed_knots"] * KNOTS_TO_MPH
    result["gust_speed_mph"] = result["gust_speed_knots"] * KNOTS_TO_MPH
    result["time_utc"] = result["hour_utc"].map(lambda x: x.isoformat())
    result["time_local"] = result["hour_utc"].dt.tz_convert(LOCAL_TZ).map(lambda x: x.isoformat())
    result["data_type"] = "observed"
    result["source"] = "IEM ASOS archive"
    result["aggregation_method"] = "latest valid report per UTC hour"
    output_cols = [
        "station_id", "station_name", "time_utc", "time_local",
        "source", "data_type", "aggregation_method", "report_count", "missing_hour",
        "last_report_utc", "temperature_f", "relative_humidity_pct",
        "wind_speed_knots", "wind_speed_mph", "wind_direction_deg",
        "precipitation_report_1h_in", "gust_speed_knots", "gust_speed_mph",
        *("missing_" + col for col in FIELDS),
    ]
    return result[output_cols]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--stations", nargs="+", metavar="ID", help="e.g. AAT BUR; default all stations")
    p.add_argument("--start-date", help="First UTC date, YYYY-MM-DD (inclusive)")
    p.add_argument("--end-date", help="Last UTC date, YYYY-MM-DD (inclusive)")
    args = p.parse_args()
    if not args.input.is_file():
        p.error(f"IEM observation CSV not found: {args.input}")
    try:
        out = normalize_iem_hourly(pd.read_csv(args.input, low_memory=False), args.stations,
                                   args.start_date, args.end_date)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out):,} hourly station records to {args.output}")
    for station, group in out.groupby("station_id"):
        print(f"  {station}: {len(group) - int(group.missing_hour.sum())}/{len(group)} hours reported; "
              f"{int(group.missing_hour.sum())} missing hours; "
              f"{int(group.report_count.sum())} raw reports")
    print("Observed station measurements only, with missing data retained as blank/NaN.")


if __name__ == "__main__":
    main()
