# EmberScope — IEM hourly normalization (next integration step)

This small addition works with the **existing** IEM downloader; it performs **no API calls** and does not change upstream source data. Only three additive files are included.

## Commands (PowerShell from repository root)

```powershell
python -m unittest discover -s tests -p "test_iem_hourly.py" -v
python .\src\data_processing\iem_hourly.py --stations AAT BUR --start-date 2026-10-01 --end-date 2026-11-07
```

Expected for the user's current raw CSV: 1,824 station-hour rows (912 each); AAT 910/912 reported hours, BUR 911/912, subject to any changes in saved data.

Output (ignored by Git): `live/california_sources/iem/iem_california_hourly.csv`

## Data contract and caveats

- One row per station per UTC hour, including hours without any reports; `missing_hour` marks the latter.
- `report_count` counts all reports within a UTC hour.
- Temperature (°F), RH (%), wind (knots and mph), direction (degrees) and gust (knots and mph) use **latest nonmissing value in the UTC hour**. This does not imply the reports were made at the top of the hour.
- `precipitation_report_1h_in` is the **latest available IEM p01i report** within that hour; **do not sum multiple p01i reports** or assume missing p01i is zero. It can represent the reported preceding 1-hour interval and is not a validated hourly total or daily gauge accumulation.
- Source collector currently parses IEM trace precipitation `T` as missing; this stage cannot recover that distinction from the saved CSV. It must be fixed upstream before calculating verified rainfall totals.
- Source/provenance retained: `source=IEM ASOS archive`, `data_type=observed`.
- All UTC slots are included; local ISO times have correct Pacific daylight-saving offsets (repeated fall-back 1 AM hours remain distinct).
- There is **no fire-risk scoring** or NFDRS computation here. Do not mix observed station series with Open-Meteo model series without explicit provenance and unit handling.

This is a data-processing foundation for an upcoming Stations dashboard view, not yet a dashboard change.
