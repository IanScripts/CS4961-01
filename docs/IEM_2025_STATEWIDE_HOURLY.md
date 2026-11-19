# EmberScope — 2025 Statewide IEM Hourly Processing

This extension **uses the existing IEM hourly normalizer** in
`src/data_processing/iem_hourly.py` and processes a large saved raw CSV with
bounded memory. It does **not** download more data, calculate wildfire risk,
or change NFDRS code.

## Why it exists

The 2025 IEM download has 161 catalog stations: 151 were marked complete by
the download manifest and 10 returned no data. Raw observations comprise
multiple reports per station-hour. Loading and normalizing ~2 million reports
for every station at once could exhaust RAM. This script streams the source
CSV once, temporarily partitions 2025 reports by station, then normalizes
one station at a time. The **manifest confirms the requested download window,
not completeness of every hour or measurement**; actual coverage is calculated
independently.

## Usage (from the repository root)

1. Check the offline tests:
   `python -m unittest discover -s tests -p "test_iem_year_batch.py" -v`
2. Build the 2025 statewide hourly dataset:
   `python .\scripts\build_iem_year_hourly.py --year 2025`
3. Inspect results:
   - `live/california_sources/iem/iem_california_hourly_2025.csv`
   - `live/california_sources/iem/iem_2025_station_quality.csv`

For 151 stations in a non-leap year, the resulting file should have
**1,322,760 station-hour slots** (151 × 8,760). Slots exist even when reports
or individual weather variables are missing. This does not imply 1,322,760
valid measurements of every variable.

The pre-existing `iem_california_hourly.csv` and the original raw observations
are **never overwritten** by the new script. The 2025 result files are written
only after all selected stations process successfully; a failure leaves
previous successful outputs alone.

## Data limitations

- Input `valid` is in UTC. The requested 2025 range is Jan 1 00:00 UTC through
  Dec 31 23:00 UTC; local California dates cross UTC calendar boundaries.
- The normalizer selects the **latest valid** reading per station UTC hour for
  each measurement. The report count is preserved. It does not interpolate.
- IEM `p01i` is a **reported one-hour precipitation accumulation**, not an
  instantaneous observation. Do **not** sum it across reports in an hour;
  the alignment and accumulation conventions need independent validation
  before direct Open-Meteo precipitation comparisons.
- ASOS observations are airport-based; a station's recorded conditions need
  not represent nearby vegetation, terrain, or a wildfire ignition site.
- Full-year download-manifest coverage means the archive request completed,
  not that hourly measurements are complete. Review the generated station
  quality report before use in ML.
- 2026 AAT/BUR experimental files remain separate. Do not mix observed station
  fields with Open-Meteo modeled weather unless explicitly labeled.

## Next steps

Audit per-field completeness; restrict ML candidate stations by measured
coverage; obtain aligned historical Open-Meteo reanalysis at catalog
coordinates as a separate source; and retain temporal/spatial leakage guards
when training a future wildfire occurrence classifier.

Generated `live/` CSV data should remain out of GitHub; commit only source,
tests, and documentation.
