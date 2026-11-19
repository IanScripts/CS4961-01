# CS4961-01 — Phase 1 integration (weather collectors + EmberScope experiment)

**Status:** Integration candidate prepared for team review. This is not a complete wildfire danger rating system and not a trained ML predictor. Only additions are in this package: no existing GitHub files are replaced.

## One owner per component; no duplicate work

| Component | Canonical owner | Phase 1 action |
|---|---|---|
| Station observations | Existing IEM and Synoptic scripts | Keep in `src/weather_apis` and validate |
| Historical / forecast weather | Existing Open-Meteo, GFS and HRRR scripts | Keep. No separate EmberScope Open-Meteo `weather.py` copied |
| Weather cleanup | Existing `src/data_processing` | Define common schema and gap-quality flags |
| Experimental baseline | EmberScope author | `src/wildfire_engine/experimental_score.py` is their original 4-feature model, unchanged |
| Historical fire geography | EmberScope author | `src/fire_data/fire_sources.py`; CAL FIRE polygons are **historical**, not active incidents or supervised ML labels by themselves |
| Backend/dashboard | Integration lead | **Do not add the second server/UI yet.** Decide one app and one dashboard before migrating its components |
| KBDI, GSI, dead/live fuel moisture and NFDRS-style indexes | Scientific team | To be implemented independently from published sources; not included here |

## Working minimum cross-project demonstration

The original repo's `src/weather_apis/open_meteo_api.py` produces `live/open_meteo_historical_socal.csv` from Open-Meteo model/reanalysis data, with UTC hourly timestamps, Fahrenheit, mph and inch units. Run that existing collector before the bridge:

```powershell
python src/weather_apis/open_meteo_api.py
python scripts/score_open_meteo_csv.py
python -m unittest discover -s tests -p 'test*.py' -v
```

The new `scripts/score_open_meteo_csv.py` **does not request weather again**. It consumes that existing CSV, converts units for the teammate's 4-feature engine, and outputs `live/experimental_score/open_meteo_daily_scores.csv`. It groups by California **local dates** including daylight saving, and skips incomplete days, missing values, duplicate timestamps and mixed forecast/historical sources. The output is explicitly *UNCALIBRATED_EXPERIMENTAL_WEATHER_SCORE*.

Its aggregation (daily max temperature, min humidity, max wind and total rainfall) is a baseline only: these extrema may not happen simultaneously. This score is neither a true fuel-moisture measure nor an NFDRS index nor a wildfire probability. It is **not suitable for operational warnings**.

To avoid cost and uncontrolled downloads, do not run historical collectors for all 161 sites until targeted tests pass. AAT and BUR sample station data should be the next integration milestone, with clear measured/modelled labels and accurate station coverage.

## Required next development before a credible ML model

1. Agree on a standardized weather and output schema: station/location ID, UTC observation time, forecast issue time when applicable, source, measured/modelled/estimated flag, units, gaps and QC. Keep daily summaries distinct from hourly records.
2. Implement the independently documented science modules, their states/initialization, calibration assumptions, and numerical tests. The example scoring module is only a comparative baseline.
3. Establish a reliable **time- and location-matched wildfire occurrence ground truth** (and carefully designed non-event examples). CAL FIRE historical perimeters and NASA thermal anomalies have important coverage and labeling limits. Prevent temporal/spatial leakage.
4. Build **one Unified Dashboard**: show actual weather, the explicitly experimental baseline, and later validated scientific measures in separate labeled panels. One server and one shared API layer.
5. For terrain/vegetation, plan USGS elevation + LANDFIRE fuel/vegetation integration once data/schema and source licensing are established. Do not add arbitrary elevation/vegetation weights to the baseline.

## Suggested seven roles for Phase 1

1. Integration lead: canonical architecture, API contracts, GitHub PR reviews, one Unified Dashboard decision and end-to-end test.
2. Observations: IEM + Synoptic; AAT/BUR sample availability, QC and provenance.
3. Weather forecasts: existing Open-Meteo/GFS/HRRR sources, forecast issue times and historical/future separation.
4. Data schema + KBDI: standardize all weather, handle time zones and missing values, research/implement drought method.
5. GSI + vegetation/live fuel moisture: independently document methods; define scientific vegetation-data inputs.
6. Dead fuel moisture: independently implement/test 1/10/100/1000-hour methods with state.
7. EmberScope author: own/validate baseline; collaborate with scientific index modules (SC/ERC/BI/IC), with Members 5/6 sharing calculation load.

**Do not copy upstream NFDRS4 source or executables.** Credit the EmberScope author for their module and maintain scientific citations; confirm professor's project reuse requirements.
