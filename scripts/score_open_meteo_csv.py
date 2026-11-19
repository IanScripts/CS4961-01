"""Bridge CS4961-01's existing Open-Meteo hourly CSV to EmberScope's
experimental 4-feature daily index. No new API client, no new weather download.

Input is Open-Meteo historical/recent CSV produced by
src/weather_apis/open_meteo_api.py, whose hourly units are °F, %, mph, inches.
Output CSV is EXPERIMENTAL and not an official NFDRS rating or fire probability.

Daily aggregates are used only for an educational baseline. They combine
extrema that may occur at different hours; they are not an official fire metric.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wildfire_engine.experimental_score import WeatherDay, score  # noqa: E402

REQUIRED = ("time", "location", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation")
NUMERIC = REQUIRED[2:]
ZONE = "America/Los_Angeles"


def expected_hours(day: pd.Timestamp) -> int:
    """Handle 23-, 24-, and 25-hour California civil days correctly."""
    start = pd.Timestamp(str(day.date()), tz=ZONE)
    next_day = pd.Timestamp(str((day + pd.Timedelta(days=1)).date()), tz=ZONE)
    return int((next_day.tz_convert("UTC") - start.tz_convert("UTC")) / pd.Timedelta(hours=1))


def create_daily_scores(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required Open-Meteo CSV columns: {', '.join(missing)}")
    work = df.copy()
    # The existing Open-Meteo collector explicitly requests timezone=UTC.
    work["timestamp_utc"] = pd.to_datetime(work["time"], errors="coerce", utc=True)
    if work["timestamp_utc"].isna().any():
        raise ValueError("Input contains invalid or missing UTC timestamps")
    for field in NUMERIC:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    if work["location"].isna().any():
        raise ValueError("Missing location label")
    work["local_date"] = work["timestamp_utc"].dt.tz_convert(ZONE).dt.strftime("%Y-%m-%d")
    has_type = "data_type" in work.columns
    has_source = "source" in work.columns
    out = []
    for (loc, date), grp in work.groupby(["location", "local_date"], sort=True):
        n = expected_hours(pd.Timestamp(date))
        # Require a single original source and data class within the scored day.
        # Do not silently mix observed, historical-modelled, and forecast hours.
        types = sorted(grp["data_type"].dropna().unique().tolist()) if has_type else ["unspecified"]
        sources = sorted(grp["source"].dropna().unique().tolist()) if has_source else ["unspecified"]
        if len(types) != 1 or len(sources) != 1:
            continue
        if len(grp) != n or grp["timestamp_utc"].nunique() != n:
            continue
        if grp[list(NUMERIC)].isna().any().any():
            continue
        # Test physical bounds rather than filling missing data with invented zeros.
        t = float(grp["temperature_2m"].max())
        rh = float(grp["relative_humidity_2m"].min())
        wind = float(grp["wind_speed_10m"].max())
        precip = float(grp["precipitation"].sum())
        weather = WeatherDay(
            day=date,
            temperature_c=(t - 32.0) * 5.0 / 9.0,
            humidity_pct=rh,
            wind_kmh=wind * 1.609344,
            rain_mm=precip * 25.4,
        )
        result = score(weather)
        out.append({
            "location": str(loc), "local_date": date,
            "source": str(sources[0]), "data_type": str(types[0]),
            "valid_hourly_records": n,
            "temp_max_f": round(t, 3), "humidity_min_pct": round(rh, 3),
            "wind_max_mph": round(wind, 3), "rain_total_in": round(precip, 4),
            "experimental_score": result["score"],
            "experimental_level": result["level"],
            "metric_kind": "UNCALIBRATED_EXPERIMENTAL_WEATHER_SCORE",
        })
    return pd.DataFrame(out, columns=["location", "local_date", "source", "data_type", "valid_hourly_records", "temp_max_f", "humidity_min_pct", "wind_max_mph", "rain_total_in", "experimental_score", "experimental_level", "metric_kind"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"live"/"open_meteo_historical_socal.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"live"/"experimental_score"/"open_meteo_daily_scores.csv")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}; first collect data with the existing Open-Meteo script")
    result = create_daily_scores(pd.read_csv(args.input, low_memory=False))
    if result.empty:
        raise SystemExit("No complete, single-source California local days found; see docs/PHASE1_INTEGRATION.md")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} EXPERIMENTAL daily scores to {args.output}")
    print("Not NFDRS and not a predicted wildfire ignition probability.")

if __name__ == "__main__":
    main()
