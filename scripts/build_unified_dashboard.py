"""Build EmberScope: one offline dashboard for independent wildfire research.

Reads the team's existing experimental scores and optional saved forecast/catalog
CSVs and statewide IEM daily observed station measurements. Does not call APIs, calculate new scores, run or import NFDRS4, or read its
legacy index files. Designed to REPLACE the existing dashboard builder (not add
another dashboard).

Run (from CS4961-01 root):
    python scripts/build_unified_dashboard.py

Optional:
    python scripts/build_unified_dashboard.py --input path/to/scores.csv \
        --forecast path/to/recent.csv --catalog path/to/iem_catalog.csv
"""
from __future__ import annotations

import argparse
import ast
from html import escape
import json
from pathlib import Path
import re
import sys
from string import Template

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dashboard.iem_daily_views import statewide_explorer_html  # EmberScope V4 observed weather daily explorer
DEFAULT_INPUT = ROOT / 'live' / 'experimental_score' / 'open_meteo_daily_scores.csv'
DEFAULT_FORECAST = ROOT / 'live' / 'open_meteo_socal.csv'
DEFAULT_CATALOG = (ROOT / 'data' / 'reference' / 'iem_california_station_catalog.csv'
                   if (ROOT / 'data' / 'reference' / 'iem_california_station_catalog.csv').is_file()
                   else ROOT / 'live' / 'california_sources' / 'iem' / 'iem_california_station_catalog.csv')
DEFAULT_IEM_DAILY = ROOT / 'live' / 'california_sources' / 'iem' / 'iem_daily_station_summary_2025.csv'
DEFAULT_IEM_QUALITY = ROOT / 'live' / 'california_sources' / 'iem' / 'iem_2025_station_quality_tiers.csv'
DEFAULT_OUTPUT = ROOT / 'emberscope.html'
OPEN_METEO_SOURCE = ROOT / 'src' / 'weather_apis' / 'open_meteo_api.py'
REQUIRED = {'location', 'local_date', 'experimental_score', 'experimental_level',
            'temp_max_f', 'humidity_min_pct', 'wind_max_mph', 'rain_total_in'}
LEVELS = ['Low', 'Moderate', 'Elevated', 'High', 'Very high']
COLORS = {'Low':'#4e8b7d','Moderate':'#7592a4','Elevated':'#e3b34b',
          'High':'#d67c51','Very high':'#b44f60'}
PALETTE = ['#227b98','#d67c51','#7770b1','#4a987e','#ac8437','#c15f89','#526c99']
PLOT_CONFIG = {'displaylogo': False, 'responsive': True, 'scrollZoom': False}


def read_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit('No experimental scores CSV at ' + str(path) +
            '\nRun: python scripts/score_open_meteo_csv.py --input live/open_meteo_historical_socal.csv')
    d = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED - set(d.columns))
    if missing: raise SystemExit('Missing score columns: ' + ', '.join(missing))
    d['local_date'] = pd.to_datetime(d['local_date'], errors='coerce')
    for col in ['experimental_score','temp_max_f','humidity_min_pct','wind_max_mph','rain_total_in']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    if d[['location','local_date','experimental_score','experimental_level']].isna().any().any():
        raise SystemExit('Missing or malformed required score values')
    if d.duplicated(['location','local_date']).any():
        raise SystemExit('Duplicate location/date scores found; resolve before plotting')
    if not d['experimental_score'].between(0,100).all():
        raise SystemExit('Experimental score outside 0–100')
    if not d['experimental_level'].isin(LEVELS).all():
        raise SystemExit('Unknown experimental category')
    if d[['temp_max_f','humidity_min_pct','wind_max_mph','rain_total_in']].isna().any().any():
        raise SystemExit('Scores contain missing numeric weather fields')
    if not d['humidity_min_pct'].between(0,100).all():
        raise SystemExit('Invalid relative humidity data')
    if (d['wind_max_mph'] < 0).any() or (d['rain_total_in'] < 0).any():
        raise SystemExit('Invalid negative wind/rain data')
    return d.sort_values(['location','local_date']).reset_index(drop=True)


def read_forecast(path: Path) -> tuple[pd.DataFrame, str]:
    required = ['time','location','temperature_2m','relative_humidity_2m','wind_speed_10m','precipitation','data_type']
    if not path.is_file():
        return pd.DataFrame(), 'No saved forecast CSV found. Run the existing Open-Meteo collector to populate this tab.'
    try:
        d = pd.read_csv(path, low_memory=False)
        if any(c not in d.columns for c in required):
            return pd.DataFrame(), 'Forecast file exists but lacks the required weather/data_type columns.'
        d = d[d['data_type'].astype(str).str.lower() == 'forecast'].copy()
        if d.empty:
            return d, 'Saved weather file contains no rows explicitly labeled forecast.'
        d['time_utc'] = pd.to_datetime(d['time'], utc=True, errors='coerce')
        for col in required[2:6]:
            d[col] = pd.to_numeric(d[col], errors='coerce')
        d = d.dropna(subset=['time_utc','location'] + required[2:6])
        d = d.drop_duplicates(['location','time_utc']).sort_values(['location','time_utc'])
        if d.empty: return d, 'Forecast data has no complete records for the plotted variables.'
        d['local_time'] = d['time_utc'].dt.tz_convert('America/Los_Angeles')
        return d, f'Saved forecast snapshot: {d["time_utc"].min():%Y-%m-%d %H:%M} to {d["time_utc"].max():%Y-%m-%d %H:%M} UTC. Not a live feed.'
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return pd.DataFrame(), 'Unable to parse optional forecast CSV: ' + str(exc)


def read_city_coords(path: Path) -> dict[str,tuple[float,float]]:
    """Read LOCATIONS literal in the existing Open-Meteo client; don't repeat values."""
    if not path.is_file(): return {}
    try:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOCATIONS' for t in node.targets):
                raw = ast.literal_eval(node.value)
                return {str(name):(float(pt[0]),float(pt[1])) for name,pt in raw.items()}
    except (ValueError, TypeError, SyntaxError, OSError, KeyError, IndexError):
        pass
    return {}


def read_catalog(path: Path) -> pd.DataFrame:
    if not path.is_file(): return pd.DataFrame()
    try:
        d = pd.read_csv(path, low_memory=False)
        if not {'station_id','latitude','longitude'}.issubset(d.columns): return pd.DataFrame()
        d['latitude'] = pd.to_numeric(d['latitude'], errors='coerce')
        d['longitude'] = pd.to_numeric(d['longitude'], errors='coerce')
        d = d.dropna(subset=['station_id','latitude','longitude'])
        d = d[d['latitude'].between(32,42.5) & d['longitude'].between(-125,-113.5)].copy()
        return d.drop_duplicates(subset=['station_id'])
    except (OSError,ValueError,pd.errors.ParserError): return pd.DataFrame()


def style(fig: go.Figure, height: int=395) -> go.Figure:
    fig.update_layout(height=height,template='plotly_white',paper_bgcolor='white',plot_bgcolor='white',
        font=dict(family='Arial, sans-serif',color='#304259',size=12),
        margin=dict(l=48,r=24,t=53,b=57),hovermode='x unified',
        legend=dict(orientation='h',y=-0.26,x=0))
    fig.update_xaxes(showgrid=True,gridcolor='#e9edf1')
    fig.update_yaxes(showgrid=True,gridcolor='#e9edf1',zeroline=False)
    return fig


def trend_chart(d: pd.DataFrame, cities:list[str]) -> go.Figure:
    fig = go.Figure()
    for i,city in enumerate(cities):
        c=d[d.location==city]
        fig.add_trace(go.Scatter(x=c.local_date,y=c.experimental_score,name=city,mode='lines',
             visible=(i==0),line=dict(color=PALETTE[i%len(PALETTE)],width=2.5),
             customdata=c.experimental_level,
             hovertemplate='%{x|%b %d, %Y}<br>Score %{y:.1f}/100 · %{customdata}<extra></extra>'))
    fig.update_layout(title='Daily experimental weather score',showlegend=False,
        yaxis=dict(title='Score / 100',range=[0,100]),xaxis=dict(title='Local calendar date'))
    return style(fig)


def weather_chart(d:pd.DataFrame,cities:list[str]) -> go.Figure:
    fig = make_subplots(specs=[[{'secondary_y':True}]])
    for i,city in enumerate(cities):
        c=d[d.location==city]
        fig.add_trace(go.Scatter(x=c.local_date,y=c.temp_max_f,name='Maximum temperature (°F)',
             visible=(i==0),mode='lines',line=dict(color='#d06947',width=2),
             hovertemplate='%{x|%b %d}<br>%{y:.1f}°F<extra></extra>'), secondary_y=False)
        fig.add_trace(go.Scatter(x=c.local_date,y=c.humidity_min_pct,name='Minimum humidity (%)',
             visible=(i==0),mode='lines',line=dict(color='#2781a2',width=2),
             hovertemplate='%{x|%b %d}<br>%{y:.1f}%<extra></extra>'), secondary_y=True)
    fig.update_layout(title='Daily temperature and relative humidity')
    fig.update_yaxes(title_text='Temperature (°F)',secondary_y=False)
    fig.update_yaxes(title_text='Humidity (%)',range=[0,100],secondary_y=True,showgrid=False)
    return style(fig)


def levels_chart(d:pd.DataFrame,cities:list[str]) -> go.Figure:
    counts=d.groupby(['location','experimental_level']).size().unstack(fill_value=0).reindex(cities).fillna(0)
    fig=go.Figure()
    for level in LEVELS:
        fig.add_trace(go.Bar(x=cities,y=counts[level].tolist() if level in counts else [0]*len(cities),
            name=level,marker_color=COLORS[level],hovertemplate='%{x}<br>%{y} days<extra>'+level+'</extra>'))
    fig.update_layout(title='Experimental classifications by location',barmode='stack',
                      yaxis=dict(title='Scored location-days'),xaxis=dict(title=''))
    return style(fig,410)


def comparison_chart(d:pd.DataFrame,cities:list[str]) -> go.Figure:
    fig=go.Figure()
    for i,city in enumerate(cities):
        c=d[d.location==city]
        fig.add_trace(go.Scatter(x=c.local_date,y=c.experimental_score,name=city,
               mode='lines',line=dict(color=PALETTE[i%len(PALETTE)],width=2),
               hovertemplate='%{x|%b %d}<br>%{y:.1f}/100<extra>'+escape(city)+'</extra>'))
    fig.update_layout(title='Compare experimental weather scores across locations',
        yaxis=dict(title='Score / 100',range=[0,100]),xaxis=dict(title='Local date'))
    return style(fig,460)


def forecast_chart(d:pd.DataFrame,cities:list[str]) -> tuple[go.Figure,list[str]]:
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    names=[]
    for i,city in enumerate(cities):
        c=d[d.location==city]
        names.append(city)
        fig.add_trace(go.Scatter(x=c.local_time,y=c.temperature_2m,name='Temperature (°F)',
            visible=(i==0),mode='lines+markers',marker=dict(size=4),
            line=dict(color='#d06947',width=2)),secondary_y=False)
        fig.add_trace(go.Scatter(x=c.local_time,y=c.relative_humidity_2m,name='Humidity (%)',
            visible=(i==0),mode='lines',line=dict(color='#2781a2',width=2)),secondary_y=True)
    fig.update_layout(title='Saved Open-Meteo forecast · temperature and humidity')
    fig.update_yaxes(title_text='Temperature (°F)',secondary_y=False)
    fig.update_yaxes(title_text='Humidity (%)',range=[0,100],secondary_y=True,showgrid=False)
    return style(fig,435),names


def forecast_wind_chart(d:pd.DataFrame,cities:list[str]) -> go.Figure:
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    for i,city in enumerate(cities):
        c=d[d.location==city]
        fig.add_trace(go.Scatter(x=c.local_time,y=c.wind_speed_10m,name='Wind speed (mph)',
            visible=(i==0),mode='lines',line=dict(color='#367f79',width=2)),secondary_y=False)
        fig.add_trace(go.Bar(x=c.local_time,y=c.precipitation,name='Precipitation (in)',
            visible=(i==0),marker_color='#758bc1',opacity=.65),secondary_y=True)
    fig.update_layout(title='Saved forecast · wind and precipitation',barmode='overlay')
    fig.update_yaxes(title_text='Wind (mph)',secondary_y=False)
    fig.update_yaxes(title_text='Precipitation (in)',secondary_y=True,showgrid=False)
    return style(fig,415)


def station_chart(citycoords:dict, catalog:pd.DataFrame, scored:set[str], observed:set[str]|None=None) -> go.Figure:
    fig=go.Figure()
    observed = observed or set()
    if not catalog.empty:
        for has_data, color, marker_size, label in [
            (False, '#6c8f9b', 6, 'IEM catalog only'),
            (True, '#318368', 11, 'IEM with saved hourly observations'),
        ]:
            subset=catalog[catalog.station_id.astype(str).isin(observed)==has_data]
            if subset.empty: continue
            texts=[]
            for _,r in subset.iterrows():
                n=escape(str(r.get('station_name','')))
                elev=escape(str(r.get('elevation_m','')))
                status='Historical observations loaded' if has_data else 'Catalog entry only; observations not confirmed'
                texts.append(f'IEM {escape(str(r.station_id))} · {n}<br>Elevation: {elev} m<br>{status}')
            fig.add_trace(go.Scatter(x=subset.longitude,y=subset.latitude,mode='markers',
                 name=f'{label} ({len(subset)})',marker=dict(size=marker_size,color=color,opacity=.85),
                 text=texts,hovertemplate='%{text}<br>Lon: %{x:.3f} · Lat: %{y:.3f}<extra></extra>'))
    if citycoords:
        names=list(citycoords)
        fig.add_trace(go.Scatter(x=[citycoords[n][1] for n in names],y=[citycoords[n][0] for n in names],
            name='Open-Meteo model coordinates',mode='markers+text',text=names,textposition='top center',
            marker=dict(size=12,color=['#d76547' if n in scored else '#adadad' for n in names],
                        line=dict(color='white',width=1.5)),
            hovertext=[n+(' · scored modeled weather point' if n in scored else ' · no score') for n in names],
            hovertemplate='%{hovertext}<br>Lon: %{x:.3f} · Lat: %{y:.3f}<extra></extra>'))
    fig.update_layout(title='California locations and optional IEM station catalog',
        xaxis=dict(title='Longitude',range=[-125,-114],scaleanchor='y',scaleratio=.83),
        yaxis=dict(title='Latitude',range=[32,42.5]),
        legend=dict(orientation='h',y=-.20,x=0))
    return style(fig,520)


def score_rows(d:pd.DataFrame,cities:list[str])->str:
    rows=[]
    for city in cities:
        c=d[d.location==city]
        high=int(c.experimental_level.isin(['High','Very high']).sum())
        rows.append(f'<tr><td><b>{escape(city)}</b></td><td>{len(c)}</td>'
                    f'<td>{c.experimental_score.mean():.1f}</td><td>{c.experimental_score.max():.1f}</td>'
                    f'<td>{high}</td><td>{c.local_date.max():%b %d, %Y}</td></tr>')
    return ''.join(rows)


def chart_html(fig:go.Figure, include_js:bool=False)->tuple[str,str]:
    html=plot(fig,output_type='div',include_plotlyjs=include_js,config=PLOT_CONFIG,auto_open=False)
    match=re.search(r'<div id="([^"]+)" class="plotly-graph-div"',html)
    if not match: raise RuntimeError('Could not identify the Plotly graph element')
    return html, match.group(1)


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--forecast',type=Path,default=DEFAULT_FORECAST)
    p.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG)
    p.add_argument('--iem-daily',type=Path,default=DEFAULT_IEM_DAILY)
    p.add_argument('--iem-quality',type=Path,default=DEFAULT_IEM_QUALITY)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    d=read_scores(args.input)
    if d.empty: raise SystemExit('Scores file contains no rows')
    cities=sorted(d.location.astype(str).unique())
    f,forecast_note=read_forecast(args.forecast)
    coords=read_city_coords(OPEN_METEO_SOURCE)
    catalog=read_catalog(args.catalog)
    observed_ids=set()  # EmberScope V4.1 all-station daily explorer (no legacy 2-station panel)
    first,score_id=chart_html(trend_chart(d,cities),include_js=True)
    weather,weather_id=chart_html(weather_chart(d,cities))
    levels,_=chart_html(levels_chart(d,cities))
    multi,_=chart_html(comparison_chart(d,cities))
    try:
        daily_html=statewide_explorer_html(args.iem_daily,args.iem_quality,observed_ids=observed_ids)
    except (OSError, ValueError) as exc:
        raise SystemExit('Invalid statewide IEM daily summary: '+str(exc)) from exc
    stations,_=chart_html(station_chart(coords,catalog,set(cities),observed_ids))
    forecast_html=''
    forecast_ids=[]
    if not f.empty:
        fp,f_names=forecast_chart(f,cities)
        fw=forecast_wind_chart(f,cities)
        fig_f,first_fc=chart_html(fp)
        fig_w,second_fc=chart_html(fw)
        forecast_html=f'<div class="cards"><article class="card wide">{fig_f}</article><article class="card wide">{fig_w}</article></div>'
        forecast_ids=[first_fc,second_fc]
    else:
        forecast_html='<div class="empty"><strong>No forecast observations to plot.</strong><p>'+escape(forecast_note)+'</p></div>'
    report_day=d.local_date.max()
    start_day=d.local_date.min()
    high_days=int(d.experimental_level.isin(['High','Very high']).sum())
    using_types=sorted(d['data_type'].dropna().astype(str).unique()) if 'data_type' in d else []
    using_sources=sorted(d['source'].dropna().astype(str).unique()) if 'source' in d else []
    provenances=', '.join(escape(x) for x in using_sources) if using_sources else 'Unspecified in score CSV'
    types=', '.join(escape(x) for x in using_types) if using_types else 'Unspecified in score CSV'
    select=''.join(f'<option value="{escape(city,quote=True)}">{escape(city)}</option>' for city in cities)
    catalog_note=(f'{len(catalog)} IEM California catalog entries. '
                  f'{len(observed_ids)} station(s) with available 2025 observed daily summaries.'
                  if not catalog.empty else 'No IEM station catalog loaded. Only existing Open-Meteo model coordinates are displayed.')
    notes=(
        '<strong>Data classification:</strong> '+types+'<br><strong>Source:</strong> '+provenances+
        '<br><strong>Scores:</strong> Daily experimental scores, not official NFDRS indexes or observed fire occurrences.'
    )
    tmpl=Template(r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EmberScope | California Wildfire Research</title>
<style>
:root{--bg:#f5f7f6;--ink:#213549;--muted:#657689;--accent:#cf6949;--edge:#dfe6e9;--teal:#1f7282}
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:1330px;margin:0 auto;padding:26px 24px 60px}.eyebrow{font-size:11px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
h1{font-size:clamp(32px,4vw,45px);margin:6px 0 7px;letter-spacing:-.04em}.subtitle{color:var(--muted);font-size:15px;margin:0 0 20px;line-height:1.6}
.notice{background:#fff8ee;color:#5e4f39;border-left:4px solid #d6a05c;padding:13px 17px;border-radius:6px;line-height:1.5;font-size:13px;margin:16px 0}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0 10px}
.station-select{border:1px solid #dfe6e9;padding:9px 11px;border-radius:8px;background:white;color:#213549}.toolbar label{font-size:13px;font-weight:bold}.toolbar select{border:1px solid var(--edge);padding:10px 12px;border-radius:8px;background:white;font-size:14px;color:var(--ink)}
.tabs{display:flex;gap:7px;flex-wrap:wrap;border-bottom:1px solid var(--edge);margin:21px 0 20px;padding-bottom:9px}
.tabs button{border:1px solid transparent;border-radius:8px;background:none;color:#53657b;cursor:pointer;padding:11px 14px;font-weight:700}
.tabs button.active{background:var(--ink);color:white}.tabs button:focus-visible{outline:3px solid var(--teal);outline-offset:2px}
.tab{display:none}.tab.active{display:block}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}
.stat,.card{background:white;border:1px solid var(--edge);border-radius:12px;padding:18px;min-width:0}.stat .num{font-size:26px;font-weight:750;font-variant-numeric:tabular-nums}
.stat .label{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.4}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.card.wide{grid-column:1 / -1}.card h2{font-size:17px;margin:0 0 8px}.small{font-size:12px;color:var(--muted);line-height:1.6}
.plotly-graph-div{width:100%!important}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin-top:14px}th,td{text-align:left;border-bottom:1px solid var(--edge);padding:12px 10px;white-space:nowrap}th{font-size:12px;color:var(--muted)}tr:last-child td{border-bottom:0}
.empty{padding:28px;background:white;border:1px dashed #b9c7d0;border-radius:12px}.empty p{line-height:1.6;color:var(--muted)}
.pills{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.pill{background:#edf3f4;color:#2d5862;border-radius:16px;padding:7px 11px;font-size:12px;font-weight:700}
.gridtext{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.source{background:#f8fafb;border:1px solid var(--edge);border-radius:9px;padding:15px}.source h3{font-size:14px;margin:0 0 7px}.source p{font-size:12px;color:var(--muted);margin:0;line-height:1.55}
footer{padding:26px 0;font-size:12px;color:var(--muted);line-height:1.65}
@media(max-width:850px){.wrap{padding:20px 13px 50px}.stats,.cards,.gridtext{grid-template-columns:1fr 1fr}.card.wide{grid-column:1/-1}}
@media(max-width:660px){.stats,.cards,.gridtext{grid-template-columns:1fr}.card.wide{grid-column:auto}.stat .num{font-size:22px}.tabs button{font-size:12px;padding:9px}}
</style></head><body><main class="wrap">
<div class="eyebrow">CS4961-01 · Combined Research Platform</div><h1>EmberScope</h1>
<p class="subtitle">One dashboard for California weather, experimental fire-weather scores, station context, historical comparisons, and saved forecasts. Built from the team's existing data files—no duplicate API requests or scoring.</p>
<div class="notice"><strong>Research prototype:</strong> The score is uncalibrated and experimental. This is not an official NFDRS index, wildfire ignition probability, live fire alert, or emergency planning tool. No legacy NFDRS calculation code or index results are included.</div>
<div class="toolbar"><label for="location">Selected location</label><select id="location">$location_options</select><span class="small" id="selection-note">Location controls the Overview and saved Forecast charts.</span></div>
<nav class="tabs" role="tablist" aria-label="EmberScope navigation">
<button type="button" role="tab" id="tabBtnOverview" aria-selected="true" class="active" data-panel="overview">Overview</button>
<button type="button" role="tab" aria-selected="false" data-panel="history">Historical</button>
<button type="button" role="tab" aria-selected="false" data-panel="forecasts">Forecasts</button>
<button type="button" role="tab" aria-selected="false" data-panel="stations">Stations</button>
<button type="button" role="tab" aria-selected="false" data-panel="compare">Comparisons</button>
<button type="button" role="tab" aria-selected="false" data-panel="sources">Sources &amp; Roadmap</button>
</nav>
<section class="tab active" id="panel-overview" role="tabpanel" aria-labelledby="tabBtnOverview">
<div class="stats"><div class="stat"><div class="num">$city_count</div><div class="label">Modeled California locations</div></div>
<div class="stat"><div class="num">$score_count</div><div class="label">Complete scored location-days</div></div>
<div class="stat"><div class="num">$high_count</div><div class="label">High / Very high experimental classifications</div></div>
<div class="stat"><div class="num">$period</div><div class="label">Scored weather period</div></div></div>
<div class="cards"><article class="card wide"><h2>Experimental score trend</h2><p class="small">The four-feature weather-score baseline; controlled by the location selector above.</p>$trend_html</article>
<article class="card wide"><h2>Weather inputs</h2><p class="small">Daily maximum temperature and minimum relative humidity; extrema can occur at different times.</p>$weather_html</article></div></section>
<section class="tab" id="panel-history" role="tabpanel"><div class="cards">
<article class="card wide"><h2>Historical classification summary</h2><p class="small">These categories reflect an uncalibrated weather formula, not observed fire occurrence.</p>$levels_html</article>
<article class="card wide"><h2>Data provenance</h2><p class="small">$source_note</p><p class="small">Daily weather scores require complete calendar days in the original scoring adapter, including daylight saving adjustments. Missing days are excluded rather than filled with zeroes.</p></article></div></section>
<section class="tab" id="panel-forecasts" role="tabpanel"><article class="card wide"><h2>Saved weather forecasts</h2><p class="small">$forecast_note</p><p class="small">Forecasts are weather variables only. We do not create future EmberScope scores from them in this dashboard.</p></article>$forecast_html</section>
<section class="tab" id="panel-stations" role="tabpanel"><div class="cards">$iem_daily_html<article class="card wide"><h2>Location and station explorer</h2><p class="small">Offline longitude/latitude locator, not a street or terrain map. Orange points: Open-Meteo model coordinates. Blue: IEM catalog-only stations. Green: IEM stations with 2025 observed daily summaries. None means a live wildfire alert.</p>$station_html</article><article class="card wide"><p class="small">$catalog_note</p><p class="small">Model query coordinates and physical IEM stations are different location types. The statewide 2025 IEM observed-weather explorer appears above. Station-derived fire-danger scores are not yet calculated.</p></article></div></section>
<section class="tab" id="panel-compare" role="tabpanel"><div class="cards"><article class="card wide"><h2>All-location comparison</h2><p class="small">Same experimental scoring method for each Open-Meteo location. Do not interpret relative scores as measured fire frequency.</p>$compare_html</article>
<article class="card wide"><h2>Location summary</h2><div class="scroll"><table><thead><tr><th>Location</th><th>Days</th><th>Mean score</th><th>Max score</th><th>High+ days</th><th>Last day</th></tr></thead><tbody>$table_rows</tbody></table></div></article></div></section>
<section class="tab" id="panel-sources" role="tabpanel"><article class="card wide"><h2>What we have vs. what the dashboard currently uses</h2>
<div class="pills"><span class="pill">Open-Meteo historical: plotted</span><span class="pill">Open-Meteo forecast: optional</span><span class="pill">IEM catalog: optional</span><span class="pill">IEM statewide 2025 daily observations: plotted</span></div>
<div class="gridtext"><div class="source"><h3>IEM ASOS/AWOS</h3><p>Existing observational collector. Statewide 2025 daily station weather and quality summaries are plotted; these are not wildfire scores.</p></div>
<div class="source"><h3>Synoptic</h3><p>Existing station-observation collector. Not plotted until validated station records are loaded.</p></div>
<div class="source"><h3>Open-Meteo</h3><p>Historical modeled weather powers experimental scores. Optional saved recent forecast weather powers the Forecasts tab.</p></div>
<div class="source"><h3>Open-Meteo GFS</h3><p>Existing longer-range forecast collector. Not included in this dashboard's current forecast chart.</p></div>
<div class="source"><h3>NOAA HRRR</h3><p>Existing short-range forecast collector. Not included in this dashboard's current forecast chart.</p></div>
<div class="source"><h3>CAL FIRE / optional NASA FIRMS</h3><p>Fire-data adapters are in the project. Neither is plotted or treated as a fire label yet.</p></div></div></article>
<article class="card wide" style="margin-top:16px"><h2>Next engine milestones</h2><div class="pills"><span class="pill">Working: experimental 4-feature score</span><span class="pill">Planned: KBDI</span><span class="pill">Planned: live/dead fuel moisture</span><span class="pill">Planned: ERC, SC, BI, IC</span><span class="pill">Future: ML with wildfire labels</span></div><p class="small">Develop the scientific modules independently from published equations with attribution and tests. The old NFDRS4 dashboard can inform interface design, but upstream NFDRS4 calculation code is not transferred.</p></article></section>
<footer>EmberScope · CS4961-01 · Generated locally from <code>$source_file</code>. Forecast and station views use other saved files only when present. No external map tiles, additional API calls, server, NFDRS4 executable, or model training needed to open this HTML.</footer>
</main><script>
(function(){
 const names=$city_json;
 const figIds=$figure_ids;
 const picker=document.getElementById('location');
 const panels=Array.from(document.querySelectorAll('.tab'));
 const buttons=Array.from(document.querySelectorAll('[data-panel]'));
 function updateLocation(){
   const i=names.indexOf(picker.value); if(i<0) return;
   if(window.Plotly){
      Plotly.restyle(figIds.score,{visible:names.map((_,j)=>j===i)});
      Plotly.restyle(figIds.weather,{visible:Array.from({length:2*names.length},(_,j)=>Math.floor(j/2)===i)});
      if(figIds.forecast.length){
          for(const id of figIds.forecast){Plotly.restyle(id,{visible:Array.from({length:2*names.length},(_,j)=>Math.floor(j/2)===i)});}
      }
   }
   document.getElementById('selection-note').textContent='Selected: '+picker.value+' · controls Overview and saved Forecast charts';
 }
 function resizeVisible(){
   if(!window.Plotly) return;
   document.querySelectorAll('.tab.active .plotly-graph-div').forEach(el=>{
      const width=Math.round(el.getBoundingClientRect().width);
      if(width>50){try{Plotly.relayout(el,{width:width})}catch(e){}}
   });
 }
 function showTab(which){
   panels.forEach(p=>p.classList.toggle('active',p.id==='panel-'+which));
   buttons.forEach(b=>{const yes=b.dataset.panel===which;b.classList.toggle('active',yes);b.setAttribute('aria-selected',yes?'true':'false')});
   requestAnimationFrame(resizeVisible);
 }
 buttons.forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.panel)));
 picker.addEventListener('change',updateLocation);
 window.addEventListener('resize',()=>requestAnimationFrame(resizeVisible));
 requestAnimationFrame(resizeVisible);
})();
</script></body></html>''')
    html=tmpl.substitute(
        location_options=select,city_count=str(len(cities)),score_count=f'{len(d):,}',
        high_count=f'{high_days:,}',period=f'{start_day:%b %d}–{report_day:%b %d, %Y}',
        trend_html=first,weather_html=weather,levels_html=levels,
        compare_html=multi,station_html=stations,forecast_html=forecast_html,
        forecast_note=escape(forecast_note),catalog_note=escape(catalog_note),source_note=notes,
        iem_daily_html=daily_html,

        table_rows=score_rows(d,cities),source_file=escape(str(args.input)),
        city_json=json.dumps(cities,ensure_ascii=False).replace('</','<\\/'),
        figure_ids=json.dumps({'score':score_id,'weather':weather_id,'forecast':forecast_ids}),
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html,encoding='utf-8')
    print(f'Built unified EmberScope dashboard: {args.output}')
    print(f'Scores: {len(d)} across {len(cities)} modeled locations; date range {start_day:%Y-%m-%d} to {report_day:%Y-%m-%d}')
    print(f'Optional forecast rows: {len(f)} | optional IEM station catalog rows: {len(catalog)}')
    print(f'IEM 2025 daily weather: {len(observed_ids)} distinct stations with summaries; the 2-station legacy panel is omitted')
    print('Experimental scores only; no NFDRS4 code, old indexes, or ML predictions included.')


if __name__=='__main__': main()
