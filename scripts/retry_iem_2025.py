"""Resume EmberScope's California IEM collection without re-downloading completed station-years.

This is a safe wrapper around src/weather_apis/iem_california.py, not a second
weather downloader. Uses the existing manifest, data schema, and CSV output.

    python scripts/retry_iem_2025.py
    python scripts/retry_iem_2025.py --year 2025 --pause-seconds 5 --max-attempts 4

A persistent 429 halts the run, preserving all completed station-years for
later recovery. Unavailable stations remain eligible for later retries.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import sys
import time
from typing import Callable, TypeVar

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


def retry_after_seconds(header: str | None, *, now: datetime | None = None) -> float | None:
    """Parse both Retry-After: seconds and HTTP date; reject invalid values."""
    if header is None or not str(header).strip():
        return None
    raw = str(header).strip()
    try:
        seconds = float(raw)
        return max(seconds, 0.0) if seconds < float("inf") else None
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            now = now or datetime.now(timezone.utc)
            return max(0.0, (when - now).total_seconds())
        except (TypeError, ValueError, OverflowError, IndexError):
            return None


def fetch_with_backoff(
    fetch: Callable[[], T], *, station: str, year: int,
    max_attempts: int = 4, base_backoff: float = 30,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    """Back off on 429, 5xx and network timeouts; stop after persistent 429."""
    if max_attempts < 1 or base_backoff < 0:
        raise ValueError("max_attempts must be positive and base_backoff nonnegative")
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch()
        except requests.RequestException as exc:
            resp = getattr(exc, "response", None)
            status = resp.status_code if resp is not None else None
            temporary = (status == 429 or
                         status is not None and 500 <= status < 600 or
                         isinstance(exc, (requests.Timeout, requests.ConnectionError)))
            if not temporary:
                raise
            if attempt >= max_attempts:
                if status == 429:
                    raise SystemExit(
                        f"IEM is still returning HTTP 429 for {station}/{year} after "
                        f"{max_attempts} attempts. Stopping requests; saved data and "
                        "manifest remain intact. Retry on a later occasion."
                    ) from exc
                raise
            retry_after = retry_after_seconds(
                resp.headers.get("Retry-After") if resp is not None else None
            )
            delay = max(base_backoff * (2 ** (attempt - 1)), retry_after or 0)
            print(f"  Temporary IEM failure for {station}/{year} "
                  f"(HTTP {status or 'network'}). Retrying after {delay:.0f}s "
                  f"[{attempt}/{max_attempts}].", flush=True)
            sleeper(delay)
    raise AssertionError("Unreachable retry state")


def remaining_stations(catalog: pd.DataFrame, manifest: pd.DataFrame,
                       year: int) -> pd.DataFrame:
    """Use completed inclusive manifest bounds to avoid redundant archive calls."""
    if "station_id" not in catalog.columns:
        raise ValueError("Station catalog lacks station_id")
    needed = {"station_id", "year", "requested_start", "requested_end"}
    if not needed.issubset(manifest.columns):
        return catalog.copy()
    starts = pd.to_datetime(manifest["requested_start"], errors="coerce")
    ends = pd.to_datetime(manifest["requested_end"], errors="coerce")
    years = pd.to_numeric(manifest["year"], errors="coerce")
    complete = manifest.loc[
        years.eq(year) &
        starts.le(pd.Timestamp(year=year, month=1, day=1)) &
        ends.ge(pd.Timestamp(year=year, month=12, day=31)),
        "station_id",
    ].astype(str).str.strip()
    return catalog.loc[~catalog["station_id"].astype(str).str.strip().isin(set(complete))].copy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2025, help="Full calendar year to finish (default 2025)")
    p.add_argument("--pause-seconds", type=float, default=5.0,
                   help="Pause between station requests, default 5 seconds")
    p.add_argument("--max-attempts", type=int, default=4,
                   help="Max attempts per station when throttled or temporarily failing")
    p.add_argument("--base-backoff", type=float, default=30,
                   help="First retry delay in seconds, doubles on later attempts")
    args = p.parse_args()
    if not 2000 <= args.year <= date.today().year:
        p.error("--year must be between 2000 and the current year")
    if args.pause_seconds < 1 or args.max_attempts < 1 or args.base_backoff < 1:
        p.error("Use at least 1 second between requests, 1 retry attempt and 1 second base backoff")

    sys.path.insert(0, str(ROOT))
    from src.weather_apis import iem_california as iem

    # Catalog and manifest already live in this repository. Do not use an
    # extra metadata API request or replace existing downloaded observations.
    if not iem.CATALOG_FILE.is_file():
        p.error(f"Missing catalog: {iem.CATALOG_FILE}. Run original IEM --catalog-only first")
    catalog = pd.read_csv(iem.CATALOG_FILE, dtype={"station_id": str}, low_memory=False)
    if iem.MANIFEST_FILE.is_file():
        manifest = pd.read_csv(iem.MANIFEST_FILE, dtype={"station_id": str}, low_memory=False)
    else:
        manifest = pd.DataFrame()

    pending = remaining_stations(catalog, manifest, args.year)
    print(f"California IEM catalog: {len(catalog)} stations")
    print(f"Manifest-complete for {args.year}: {len(catalog) - len(pending)}")
    print(f"Still eligible for {args.year}: {len(pending)}")
    if pending.empty:
        print("No station-years left to request.")
        return

    original_fetch = iem.fetch_station_year

    def guarded_fetch(station: str, year: int) -> pd.DataFrame:
        return fetch_with_backoff(
            lambda: original_fetch(station, year),
            station=station, year=year,
            max_attempts=args.max_attempts,
            base_backoff=args.base_backoff,
        )

    iem.fetch_station_year = guarded_fetch
    iem.PAUSE_SECONDS = args.pause_seconds
    try:
        new_rows, failed = iem.download_observations(
            pending, date(args.year, 1, 1), date(args.year, 12, 31)
        )
    except KeyboardInterrupt:
        print("Interrupted. Existing observations and manifest entries were preserved.")
        raise SystemExit(130)
    print(f"New observations saved: {new_rows:,}")
    print(f"Unresolved station requests: {failed}")
    print(f"Observations: {iem.OBS_FILE}")
    print(f"Manifest: {iem.MANIFEST_FILE}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
