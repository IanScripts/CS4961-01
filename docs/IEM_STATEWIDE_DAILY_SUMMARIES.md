# EmberScope statewide daily IEM summaries

Run from the CS4961-01 project root:

```powershell
python -m unittest discover -s tests -p "test_iem_statewide_daily.py" -v
python .\scripts\build_iem_statewide_daily.py --year 2025
```

Input: `live/california_sources/iem/iem_california_hourly_2025.csv` (local ~293 MB; **do not commit it**).

Output: `live/california_sources/iem/iem_daily_station_summary_2025.csv` (station-local daily summaries intended for compact EmberScope V4 charts; **do not commit it**).

The input is read in chunks. The script calculates daily mean and maximum temperatures, mean and minimum humidity, mean and maximum wind speed, plus counts of hourly observations for each variable. It keeps missingness visible via count columns; it **does not** replace missing data with zero. The output contains only complete California calendar-day **hourly slots** (23/24/25-hour DST-adjusted days), not necessarily complete measurements. Every value is observed weather, not an NFDRS calculation or fire probability.

**Calendar-boundary caveat:** The 2025 station-hour file spans January 1, 2025 00:00 UTC through January 1, 2026 00:00 UTC. This cuts across California local midnight, so the December 31, 2025 *local* date lacks its final eight hours. The script excludes that incomplete local day rather than displaying a misleading daily summary, as well as the partial December 31, 2024 local day. It should produce 364 complete local-day grids per station (January 1 through December 30, 2025), assuming every input station has its full UTC-year hourly grid.

The script deliberately avoids summing IEM `p01i` into daily precipitation: report validity windows/overlap must be validated before interpreting daily totals. It only counts hourly slots with a precipitation report. This summary is for visualization, not model training; preserve the original per-hour observations for supervised ML and weather model evaluation.

Use the small `iem_2025_station_quality_tiers.csv` (already created) to filter station-level quality. The complete hourly data is kept locally for on-demand inspection. Do not pass the 293MB source straight to the existing V3 `--iem-hourly` builder; it creates per-station traces and an excessively large self-contained dashboard.
