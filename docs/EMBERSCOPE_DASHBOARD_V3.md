# EmberScope V3 — observed station weather in the existing dashboard

V3 updates **only** the existing `scripts/build_unified_dashboard.py` and adds
`src/dashboard/iem_views.py`, along with automated tests. It does not create
another dashboard, call APIs, alter the IEM raw data, change the source
collectors, run NFDRS4, or calculate fire-danger indices.

## Prerequisite

The processed IEM hourly CSV must already exist at:

`live/california_sources/iem/iem_california_hourly.csv`

Create from existing archived reports, from the repository root:

```powershell
python .\src\data_processing\iem_hourly.py --stations AAT BUR --start-date 2026-10-01 --end-date 2026-11-07
```

The CSV is **local generated data** and is not included in this release/ZIP.
Don't upload large weather archives or generated dashboard HTML to GitHub.

## Test and build

```powershell
python -m unittest discover -s tests -p "test_iem_dashboard.py" -v
python .\scripts\build_unified_dashboard.py
Start-Process chrome.exe -ArgumentList "file:///C:/Users/ianbp/Downloads/CS4961-01-git/emberscope.html"
```

Or supply a custom path: `--iem-hourly path/to/iem_california_hourly.csv`.

## Stations view

- IEM catalog dots with downloaded historical observations are marked **green**;
  catalog-only records remain **blue**. Model query coordinates remain orange.
- A **separate observed-station selector** appears in the Stations tab because
  the global six-location selector controls the *modeled* Open-Meteo score and
  forecast charts, not sensor observations.
- Historical IEM hourly temperature (°F), relative humidity (%), wind speed
  (mph), and IEM `p01i` hourly precipitation reports (inches) are displayed.
- Reported hours and measurement availability are shown per station; NaN
  measurements **remain gaps**. No zero-fill or interpolation is applied.
- Precipitation is plotted as the retained hourly report, **not summed**:
  overlapping one-hour reports cannot safely be treated as independent totals.
- All traces use California local timestamps with offset so both repeated DST
  01:00 hours remain distinct. The source data itself is UTC.

## Research limitations

Saved IEM data are observational station reports and not live conditions.
They may have instrument or archive errors and should be validated for
scientific use. The current daily four-feature experimental weather score
is sourced from **Open-Meteo modeled weather only** and is not computed for
IEM stations. No official NFDRS indices, ignition probabilities, ML models,
or validated comparisons are implied. The V3 addition uses no NFDRS4 code.

## Files

- `scripts/build_unified_dashboard.py` — replacement dashboard generator
- `src/dashboard/__init__.py` — Python package marker
- `src/dashboard/iem_views.py` — offline station data parsing and Plotly charts
- `tests/test_iem_dashboard.py` — unit tests
- `docs/EMBERSCOPE_DASHBOARD_V3.md` — this document
