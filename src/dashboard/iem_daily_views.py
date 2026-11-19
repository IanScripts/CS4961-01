"""Original EmberScope statewide IEM weather-view adapter.

Consumes precomputed observational day summaries. Produces a self-contained,
select-on-demand chart section for the existing dashboard. No NFDRS4 imports,
calculation code, wildfire predictions, or remote requests.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "station_id", "local_date", "day_timezone", "source", "data_type",
    "expected_local_hours", "hour_slots", "hours_with_reports", "report_coverage_pct",
    "temperature_hours", "temperature_mean_f", "temperature_max_f",
    "humidity_hours", "humidity_mean_pct", "humidity_min_pct",
    "wind_hours", "wind_mean_mph", "wind_max_mph", "precipitation_report_hours",
}
METRICS = (
    "temperature_mean_f", "temperature_max_f", "humidity_mean_pct",
    "humidity_min_pct", "wind_mean_mph", "wind_max_mph", "report_coverage_pct",
)
TIERS = ("Strong", "Good", "Moderate", "Limited")


def _read_tiers(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.is_file():
        return {}, {}
    q = pd.read_csv(path, dtype={"station_id": "string"}, low_memory=False)
    if "station_id" not in q.columns:
        raise ValueError(f"Quality tiers file has no station_id column: {path}")
    tier_cols = ("quality_tier", "tier", "station_quality_tier", "coverage_tier", "quality_class", "classification")
    tier_col = next((col for col in tier_cols if col in q.columns), None)
    if tier_col is None:
        # Leave station quality unknown rather than claiming a tier we can't justify.
        return {}, {}
    if q["station_id"].duplicated().any():
        raise ValueError("Duplicate station IDs in quality tiers file")
    names = next((c for c in ("station_name", "name", "site_name") if c in q.columns), None)
    mapping: dict[str, str] = {}
    station_names: dict[str, str] = {}
    for row in q.to_dict("records"):
        sid = str(row["station_id"]).strip()
        level = str(row[tier_col]).strip()
        matched = next((t for t in TIERS if level.casefold() == t.casefold()), None)
        mapping[sid] = matched or (level if level and level.lower() != "nan" else "Unclassified")
        if names and pd.notna(row[names]):
            station_names[sid] = str(row[names]).strip()
    return mapping, station_names


def _numeric_or_none(value: object, *, decimals: int = 2):
    if pd.isna(value):
        return None
    return round(float(value), decimals)


def _validated_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"station_id": "string"}, low_memory=False)
    missing = sorted(REQUIRED - set(d.columns))
    if missing:
        raise ValueError("Statewide daily summary missing columns: " + ", ".join(missing))
    if d.empty:
        raise ValueError("Statewide daily summary is empty")
    if d["station_id"].isna().any() or d["local_date"].isna().any():
        raise ValueError("Daily summary has missing station IDs or local dates")
    if d.duplicated(["station_id", "local_date"]).any():
        raise ValueError("Duplicate station/date rows in statewide daily summary")
    if not d["data_type"].eq("observed").all():
        raise ValueError("Daily summary contains records not labeled observed")
    if not d["source"].eq("IEM ASOS archive").all():
        raise ValueError("Daily summary has unexpected weather provenance")
    if not d["day_timezone"].eq("America/Los_Angeles").all():
        raise ValueError("Unexpected daily summary timezone")
    dates = pd.to_datetime(d["local_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid local_date in summary")
    d["local_date"] = dates.dt.strftime("%Y-%m-%d")
    count_cols = ("expected_local_hours", "hour_slots", "hours_with_reports", "temperature_hours", "humidity_hours", "wind_hours", "precipitation_report_hours")
    for col in (*count_cols, *METRICS):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d[list(count_cols)].isna().any().any():
        raise ValueError("Invalid hourly slot/count value in daily summary")
    if not d["expected_local_hours"].isin([23, 24, 25]).all():
        raise ValueError("Daily summary contains invalid California local-day hour count")
    if not d["hour_slots"].eq(d["expected_local_hours"]).all():
        raise ValueError("Daily summary contains incomplete local-day grids")
    for col in count_cols[2:]:
        if ((d[col] < 0) | (d[col] > d["hour_slots"])).any():
            raise ValueError(f"Invalid report count in {col}")
    for col in ("humidity_mean_pct", "humidity_min_pct", "report_coverage_pct"):
        bad = d[col].notna() & ~d[col].between(0, 100)
        if bad.any():
            raise ValueError(f"Value out of 0–100 range in {col}")
    for col in ("wind_mean_mph", "wind_max_mph"):
        if (d[col].dropna() < 0).any():
            raise ValueError(f"Negative wind value in {col}")
    return d.sort_values(["station_id", "local_date"])


def statewide_explorer_html(daily_path: Path, quality_path: Path, *, observed_ids: set[str] | None = None) -> str:
    """HTML with one group of charts dynamically redrawn for the selected station."""
    if not daily_path.is_file():
        return (
            '<article class="card wide"><h2>2025 Statewide Weather Explorer</h2>'
            '<p class="small">Daily IEM summaries not provided; generate '
            '<code>iem_daily_station_summary_2025.csv</code> to enable this view.</p></article>'
        )
    d = _validated_daily(daily_path)
    quality, names = _read_tiers(quality_path)
    stations = {}
    for sid, part in d.groupby("station_id", sort=True):
        sid = str(sid)
        stations[sid] = {
            "quality": quality.get(sid, "Unclassified"),
            "name": names.get(sid, sid),
            "dates": part["local_date"].tolist(),
            "expected": int(part["hour_slots"].sum()),
            "observed": int(part["hours_with_reports"].sum()),
            "temperature_hours": int(part["temperature_hours"].sum()),
            "humidity_hours": int(part["humidity_hours"].sum()),
            "wind_hours": int(part["wind_hours"].sum()),
            "series": {col: [_numeric_or_none(v) for v in part[col].tolist()] for col in METRICS},
        }
    if observed_ids is not None:
        observed_ids.update(stations)
    payload = json.dumps({"stations": stations, "days": len(d)}, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    payload = payload.replace("</", "<\\/")
    return (
        '<article class="card wide" id="v4-explorer">'
        '<h2>2025 California Statewide Weather Explorer</h2>'
        f'<p class="small">IEM ASOS archive observations, {len(stations)} stations, '
        f'{len(d):,} complete station-local-day hourly grids. '
        'Dates shown in America/Los_Angeles. Measurement coverage can be incomplete even on a full hourly grid. '
        'No estimated values, fire-danger indexes, or wildfire probabilities.</p>'
        '<div class="toolbar"><label for="v4-quality">Station quality</label>'
        '<select id="v4-quality" class="station-select"><option value="All">All stations</option></select>'
        '<label for="v4-station">Station</label>'
        '<select id="v4-station" class="station-select"></select></div>'
        '<p class="small" id="v4-station-summary">Choose a station to view recorded weather.</p>'
        '<div id="v4-temperature" style="min-height:300px"></div>'
        '<div id="v4-humidity" style="min-height:300px"></div>'
        '<div id="v4-wind" style="min-height:300px"></div>'
        '<div id="v4-coverage" style="min-height:260px"></div>'
        '<p class="small">Coverage means hours with station reports, not necessarily valid readings for every variable. '
        'Wind maxima are daily weather observations. Precipitation report counts are not daily totals.</p>'
        '</article>'
        '<script type="application/json" id="v4-data">' + payload + '</script>\n'
        '<script>\n' + _EXPLORER_JS + '\n</script>'
    )


_EXPLORER_JS = r"""
(() => {
 'use strict';
 const parsed = JSON.parse(document.getElementById('v4-data').textContent);
 const stations = parsed.stations;
 const ids = Object.keys(stations).sort();
 const qualitySelect = document.getElementById('v4-quality');
 const stationSelect = document.getElementById('v4-station');
 const summary = document.getElementById('v4-station-summary');
 const qualities = Array.from(new Set(ids.map(id => stations[id].quality))).sort((a,b) =>
   ['Strong','Good','Moderate','Limited','Unclassified'].indexOf(a) -
   ['Strong','Good','Moderate','Limited','Unclassified'].indexOf(b));
 qualities.forEach(q => { const opt=document.createElement('option'); opt.value=q; opt.textContent=q; qualitySelect.append(opt); });
 function rebuildStations() {
   const selected = stationSelect.value;
   stationSelect.replaceChildren();
   ids.filter(id => qualitySelect.value === 'All' || stations[id].quality === qualitySelect.value).forEach(id => {
      const opt=document.createElement('option'); opt.value=id;
      opt.textContent=id + (stations[id].name !== id ? ' - ' + stations[id].name : '');
      stationSelect.append(opt);
   });
   if (Array.from(stationSelect.options).some(o => o.value===selected)) stationSelect.value=selected;
   if (!stationSelect.value) summary.textContent='No stations in selected quality tier.';
   renderIfVisible();
 }
 function trace(s, key, title, color) { return {type:'scatter',mode:'lines',name:title,x:s.dates,y:s.series[key],line:{color:color,width:1.8},connectgaps:false,hovertemplate:'%{x}<br>%{y:.2f}<extra>'+title+'</extra>'}; }
 function chart(id, rows, heading, unit, minVal, maxVal) {
   const layout={title:{text:heading,font:{size:14}},margin:{l:62,r:18,t:45,b:50},height:id==='v4-coverage'?260:310,
     paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',hovermode:'x unified',
     xaxis:{title:'2025 station-local date',type:'date'},yaxis:{title:unit,automargin:true},legend:{orientation:'h',y:-0.28}};
   if (minVal!==null) layout.yaxis.range=[minVal,maxVal];
   Plotly.react(id, rows, layout, {responsive:true,displaylogo:false,scrollZoom:false});
 }
 function renderCharts() {
   if (!stationSelect.value || !window.Plotly) return;
   const id=stationSelect.value, s=stations[id];
   summary.textContent=id + ' | ' + s.name + ' | ' + s.quality +
     ' | ' + s.dates.length + ' local dates | ' + s.observed + '/' + s.expected + ' hours with reports' +
     ' | temperature hours: '+s.temperature_hours+' | humidity hours: '+s.humidity_hours+' | wind hours: '+s.wind_hours;
   chart('v4-temperature', [trace(s,'temperature_mean_f','Daily mean','#227b98'),trace(s,'temperature_max_f','Daily maximum','#d67c51')], 'Observed temperature', '°F', null, null);
   chart('v4-humidity', [trace(s,'humidity_mean_pct','Daily mean','#227b98'),trace(s,'humidity_min_pct','Daily minimum','#d67c51')], 'Observed relative humidity','%',0,100);
   chart('v4-wind', [trace(s,'wind_mean_mph','Daily mean','#227b98'),trace(s,'wind_max_mph','Daily maximum','#d67c51')], 'Observed wind speed','mph',null,null);
   chart('v4-coverage',[trace(s,'report_coverage_pct','Hours with reports','#4a987e')], 'Reported hourly coverage','%',0,100);
 }
 function renderIfVisible() {
   if (document.getElementById('panel-stations').classList.contains('active')) {
     window.requestAnimationFrame(renderCharts);
   }
 }
 qualitySelect.addEventListener('change', rebuildStations);
 stationSelect.addEventListener('change', renderIfVisible);
 // Register before the builder's existing tab handler, then draw next frame.
 document.querySelectorAll('[data-panel="stations"]').forEach(btn => btn.addEventListener('click', () =>
   window.requestAnimationFrame(renderCharts)));
 rebuildStations();
})();
"""
