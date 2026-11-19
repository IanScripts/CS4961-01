"""Stream one calendar year of IEM reports into standardized station-hour CSV.

Uses the existing, independently tested normalize_iem_hourly() logic.  Reads
large report files in chunks, partitions one year of completed stations into
temporary per-station files, then processes only one station at a time.

No web requests, fire danger calculations, NFDRS source, or missing-value filling.
Only stations marked full-year complete in the IEM download manifest are used.

From the CS4961-01 project root:
    python scripts/build_iem_year_hourly.py --year 2025

For a smaller check:
    python scripts/build_iem_year_hourly.py --year 2025 --stations AAT BUR \
        --output live/california_sources/iem/iem_hourly_2025_pilot.csv \
        --quality-output live/california_sources/iem/iem_quality_2025_pilot.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing.iem_hourly import normalize_iem_hourly  # noqa: E402

IEM_DIR = ROOT / "live" / "california_sources" / "iem"
DEFAULT_INPUT = IEM_DIR / "iem_california_observations.csv"
DEFAULT_MANIFEST = IEM_DIR / "iem_download_manifest.csv"
DEFAULT_CATALOG = IEM_DIR / "iem_california_station_catalog.csv"



def completed_stations(manifest: pd.DataFrame, catalog: pd.DataFrame, year: int) -> tuple[list[str], list[str]]:
    """Return catalog-ordered complete and incomplete station IDs for year."""
    for name, columns in (("manifest", ("station_id", "year", "requested_start", "requested_end")),
                          ("catalog", ("station_id",))):
        source = manifest if name == "manifest" else catalog
        missing = set(columns) - set(source.columns)
        if missing:
            raise ValueError(f"{name} lacks: {', '.join(sorted(missing))}")

    start, end = date(year, 1, 1), date(year, 12, 31)
    valid = manifest.copy()
    valid["station_id"] = valid["station_id"].astype("string").str.strip()
    valid["year"] = pd.to_numeric(valid["year"], errors="coerce")
    # Invalid or incomplete manifest dates are not treated as completed.
    valid["requested_start"] = pd.to_datetime(valid["requested_start"], errors="coerce").dt.date
    valid["requested_end"] = pd.to_datetime(valid["requested_end"], errors="coerce").dt.date
    valid = valid[(valid["year"] == year)
                  & (valid["requested_start"] <= start)
                  & (valid["requested_end"] >= end)]
    complete = set(valid["station_id"].dropna().tolist())
    catalog_ids = list(dict.fromkeys(catalog["station_id"].astype("string").dropna().str.strip()))
    included = [sid for sid in catalog_ids if sid in complete]
    excluded = [sid for sid in catalog_ids if sid not in complete]
    return included, excluded


def quality_row(station: str, hourly: pd.DataFrame, year: int) -> dict:
    total = len(hourly)
    reported = int((hourly["report_count"] > 0).sum())
    out = {
        "station_id": station,
        "year": year,
        "total_hour_slots": total,
        "hours_with_reports": reported,
        "hours_no_reports": total - reported,
        "report_coverage_pct": round(100 * reported / total, 2) if total else 0.0,
        "raw_reports_in_hours": int(hourly["report_count"].sum()),
        "data_type": "observed",
        "source": "IEM ASOS archive",
    }
    for key, col in (
        ("temperature", "temperature_f"),
        ("humidity", "relative_humidity_pct"),
        ("wind_speed", "wind_speed_mph"),
        ("precipitation_report", "precipitation_report_1h_in"),
    ):
        n = int(hourly[col].notna().sum())
        out[f"{key}_hours"] = n
        out[f"{key}_coverage_pct"] = round(100 * n / total, 2) if total else 0.0
    return out


def build(
    input_file: Path, manifest_file: Path, catalog_file: Path,
    output_file: Path, quality_file: Path, year: int,
    stations: list[str] | None = None, chunk_size: int = 75_000,
) -> tuple[int, int]:
    if not 1900 <= year <= 2100:
        raise ValueError("Year must be between 1900 and 2100")
    if chunk_size < 100:
        raise ValueError("--chunk-size must be at least 100")
    if output_file.resolve() == input_file.resolve():
        raise ValueError("Output may not overwrite raw observations")
    if output_file.resolve() == quality_file.resolve():
        raise ValueError("Hourly and quality outputs must differ")
    for path in (input_file, manifest_file, catalog_file):
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")

    catalog = pd.read_csv(catalog_file, dtype={"station_id": "string"}, low_memory=False)
    manifest = pd.read_csv(manifest_file, dtype={"station_id": "string"}, low_memory=False)
    complete, incomplete = completed_stations(manifest, catalog, year)
    if stations:
        requested = set(stations)
        invalid = requested - set(complete)
        if invalid:
            raise ValueError("Requested station(s) are not manifest-complete: " + ", ".join(sorted(invalid)))
        complete = [sid for sid in complete if sid in requested]
    if not complete:
        raise ValueError(f"No stations marked complete for {year}")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,16}", sid) for sid in complete):
        raise ValueError("Station ID contains characters unsafe for temporary filename")

    print(f"Catalog: {len(catalog)} entries; complete for {year}: {len(complete)} selected")
    if not stations:
        print(f"Not manifest-complete: {len(incomplete)} ({', '.join(incomplete) if incomplete else 'none'})")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    quality_file.parent.mkdir(parents=True, exist_ok=True)
    # Stage all outputs inside the target directory; do not overwrite the previous
    # successful output unless every selected station finishes successfully.
    with tempfile.TemporaryDirectory(prefix="iem_year_stage_", dir=output_file.parent) as temp:
        temp_dir = Path(temp)
        csv_parts = temp_dir / "parts"
        csv_parts.mkdir()
        selected_set = set(complete)
        total_raw = 0
        seen = set()
        for idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size,
                                                 dtype={"station_id": "string"}, low_memory=False), start=1):
            if not {"station_id", "valid"}.issubset(chunk.columns):
                raise ValueError("Raw IEM CSV must contain station_id and valid columns")
            chunk = chunk.loc[chunk["station_id"].isin(selected_set)
                              & chunk["valid"].astype("string").str.startswith(f"{year}-", na=False)]
            for station, subset in chunk.groupby("station_id", sort=False):
                part = csv_parts / f"{station}.csv"
                subset.to_csv(part, mode="a", index=False, header=not part.exists())
                seen.add(str(station))
                total_raw += len(subset)
            if idx % 10 == 0:
                print(f"Read {idx * chunk_size:,} raw rows; selected {total_raw:,} reports")

        absent = set(complete) - seen
        if absent:
            raise ValueError("Manifest says complete but raw reports are absent: " + ", ".join(sorted(absent)))

        start = f"{year}-01-01"
        end = f"{year}-12-31"
        slots = int((date(year + 1, 1, 1) - date(year, 1, 1)) / timedelta(hours=1))
        staged_hourly = temp_dir / "hourly.csv"
        staged_quality = temp_dir / "quality.csv"
        quality = []
        wrote = False
        for index, station in enumerate(complete, start=1):
            reports = pd.read_csv(csv_parts / f"{station}.csv", low_memory=False,
                                  dtype={"station_id": "string"})
            try:
                hourly = normalize_iem_hourly(reports, stations=[station],
                                              start_date=start, end_date=end)
            except Exception as exc:
                raise RuntimeError(f"Hourly normalization failed for {station}: {exc}") from exc
            if len(hourly) != slots or hourly["time_utc"].nunique() != slots:
                raise ValueError(f"Incorrect {year} hourly grid for {station}")
            if int(hourly["report_count"].sum()) != len(reports):
                raise ValueError(f"Report count mismatch for {station}")
            hourly.to_csv(staged_hourly, mode="a", header=not wrote, index=False)
            wrote = True
            quality.append(quality_row(station, hourly, year))
            print(f"[{index}/{len(complete)}] {station}: "
                  f"{int((hourly.report_count > 0).sum()):,}/{slots:,} observed hours")

        pd.DataFrame(quality).to_csv(staged_quality, index=False)
        # Atomically replace final files only after successful complete processing.
        os.replace(staged_hourly, output_file)
        os.replace(staged_quality, quality_file)

    print(f"Saved {len(complete) * slots:,} station-hour rows across {len(complete)} stations")
    print(f"Hourly file: {output_file}")
    print(f"Quality report: {quality_file}")
    print("Historical observed measurements only; missing data preserved as blank/NaN.")
    return len(complete), len(complete) * slots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--output", type=Path)
    p.add_argument("--quality-output", type=Path)
    p.add_argument("--stations", nargs="+", help="Optional completed station IDs for a pilot")
    p.add_argument("--chunk-size", type=int, default=75_000)
    args = p.parse_args()
    if args.stations and (args.output is None or args.quality_output is None):
        p.error("With --stations, provide both --output and --quality-output (to avoid overwriting statewide results)")
    output = args.output or (IEM_DIR / f"iem_california_hourly_{args.year}.csv")
    quality = args.quality_output or (IEM_DIR / f"iem_{args.year}_station_quality.csv")
    try:
        build(args.input, args.manifest, args.catalog, output, quality,
              args.year, args.stations, args.chunk_size)
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
        p.exit(1, f"ERROR: {exc}\nExisting successful outputs were not overwritten.\n")


if __name__ == "__main__":
    main()
