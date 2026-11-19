# EmberScope — Unified Dashboard integration (v2)

This addition **updates the existing dashboard builder** instead of creating a second
application. It combines the EmberScope score view with selected ideas from the
older research dashboard: historical views, forecast charts, a station catalog
explorer, cross-location comparison, and source/method transparency.

## Contents

- `scripts/build_unified_dashboard.py` — replacement for the previous builder.
  It reads current local CSVs only. It never makes API requests, recomputes scores,
  trains models, invokes NFDRS4 or imports any legacy NFDRS4 library.
- `data/reference/iem_california_station_catalog.csv` — optional *data* snapshot
  copied from the user's earlier IEM CA_ASOS catalog download (retrieved
  November 6, 2026), **not a library or NFDRS4 code**. Catalog entries show
  available station metadata, **not proof of observation downloads**.

## On Windows PowerShell, from `CS4961-01-git`

1. **Backup** your previous dashboard builder outside the repo:
   `Copy-Item .\scripts\build_unified_dashboard.py "$env:USERPROFILE\Downloads\build_unified_dashboard_before_merge.py" -Force`
2. Extract `EmberScope_Unified_Merge_V2.zip` to the cloned repository root:
   `Expand-Archive -Path "$env:USERPROFILE\Downloads\EmberScope_Unified_Merge_V2.zip" -DestinationPath . -Force`
3. **Generate** the new version from the 534 scores you already computed:
   `python .\scripts\build_unified_dashboard.py`
4. **Open** it:
   `Start-Process chrome.exe -ArgumentList (Resolve-Path .\emberscope.html).Path`
5. **Check** git changes before committing:
   `git status --short`

Requires `pandas` and `plotly` (already used by the existing project).

## Tabs

- **Overview:** Existing four-feature experimental score trend and weather input chart.
- **Historical:** Historical category distributions and recorded input provenance.
- **Forecasts:** Optional saved Open-Meteo forecast **weather**, no forecasted fire score.
- **Stations:** Offline longitude/latitude locator with model query coordinates
  and cataloged IEM stations (without external map tiles).
- **Comparisons:** Cross-location score trends and summary table.
- **Sources & Roadmap:** Five weather sources, actual current use vs. available
  collectors, and as-yet-unimplemented research models.

One **global location selector** controls Overview and Forecast figures. All
charts use inline Plotly JS, so `emberscope.html` works offline.

## Limitations / originality

The figures are **not** official NFDRS indices, wildfire ignition probabilities,
real-time fire detection, or operational fire danger forecasts. All points shown
for the six cities are Open-Meteo model query coordinates, **not observed stations**.
No code from the upstream NFDRS4 engine is transferred. The legacy NFDRS4
results are intentionally excluded; fuel moisture, KBDI, GSI, ERC, SC, BI,
IC and machine-learning evaluation still require original implementation and
scientific validation.

The IEM catalog is third-party observation metadata: acknowledge Iowa
Environmental Mesonet in any published work. Keep credentials out of the repo.
