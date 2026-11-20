"""Export existing AI predictions to an offline preview and a frontend data contract.

Run from the CS4961-01 repo root (does NOT train a model):
    python -m scripts.export_ai_dashboard

Requires: pandas, geopandas, pyproj, plotly, shapely (already used by the AI pipeline).
Default inputs: data/ai/weather_baseline_v1/{map_predictions_2021_12.csv,metrics.json}
                wildfire-datasets/boundaries/counties_2024.zip
Outputs:       data/ai/weather_baseline_v1/{ai_map_preview.html,ai_map_bundle.json}
"""
from __future__ import annotations

import argparse
import json
import os
from html import escape
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pyproj import Transformer

PREDICTIONS = Path('data/ai/weather_baseline_v1/map_predictions_2021_12.csv')
METRICS = Path('data/ai/weather_baseline_v1/metrics.json')
BOUNDARIES = Path('wildfire-datasets/boundaries/counties_2024.zip')
OUT = Path('data/ai/weather_baseline_v1')
COUNTIES = {'06037': 'Los Angeles', '06111': 'Ventura', '06083': 'Santa Barbara',
            '06059': 'Orange', '06065': 'Riverside', '06071': 'San Bernardino',
            '06073': 'San Diego', '06025': 'Imperial', '06029': 'Kern'}
CRS = 'EPSG:5070'
CELL_SIZE = 10_000


def checked_input(predictions: Path, metrics_path: Path):
    required = {'grid_cell', 'target_date_local', 'model_probability', 'fire_count'}
    data = pd.read_csv(predictions, dtype={'grid_cell': str, 'target_date_local': str})
    if not required.issubset(data.columns):
        raise ValueError(f'Prediction CSV missing columns: {sorted(required - set(data.columns))}')
    if data.empty:
        raise ValueError('Prediction CSV is empty')
    if data[['grid_cell', 'target_date_local', 'fire_count']].isna().any(axis=1).any():
        raise ValueError('Missing cell, target date or fire count')
    if data.duplicated(['grid_cell', 'target_date_local']).any():
        raise ValueError('Duplicate grid-cell / target-date row')
    if not data.grid_cell.str.fullmatch(r'-?\d+_-?\d+').all():
        raise ValueError('Invalid grid cell ID')
    parsed = pd.to_datetime(data.target_date_local, format='%Y-%m-%d', errors='raise')
    if parsed.dt.strftime('%Y-%m-%d').ne(data.target_date_local).any():
        raise ValueError('Invalid target date')
    data['model_probability'] = pd.to_numeric(data.model_probability, errors='coerce')
    probabilities = data.model_probability.dropna()
    if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
        raise ValueError('Model scores must be probabilities between 0 and 1')
    counts = pd.to_numeric(data.fire_count, errors='raise')
    if not np.isfinite(counts).all() or not (counts >= 0).all() or not (counts % 1 == 0).all():
        raise ValueError('Invalid fire counts')
    data['fire_count'] = counts.astype(int)
    if 'weather_date_gridmet' in data.columns:
        lag = (parsed - pd.to_datetime(data.weather_date_gridmet, errors='raise')).dt.days
        if not lag.eq(2).all():
            raise ValueError('Weather dates must be two calendar days before target dates')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    validation = metrics.get('validation', {})
    needed = ('scored_rows', 'positive_rate', 'average_precision', 'roc_auc', 'top_1pct')
    if any(name not in validation for name in needed):
        raise ValueError('metrics.json lacks required validation metrics')
    return data, metrics


def load_counties(boundary_path: Path):
    counties = gpd.read_file(boundary_path)
    if 'GEOID' not in counties or counties.crs is None:
        raise ValueError('Census boundaries require GEOID and a declared CRS')
    counties['GEOID'] = counties.GEOID.astype(str).str.zfill(5)
    counties = counties[counties.GEOID.isin(COUNTIES)].copy()
    if set(counties.GEOID) != set(COUNTIES) or counties.GEOID.duplicated().any():
        raise ValueError('Missing or duplicate California county polygons')
    return counties.to_crs(CRS).sort_values('GEOID')


def geometry_rings(geom):
    """Exterior boundaries only; county labels are reference overlays, not spatial tests."""
    if geom.geom_type == 'Polygon':
        return [geom.exterior]
    if geom.geom_type == 'MultiPolygon':
        return [part.exterior for part in geom.geoms]
    raise ValueError(f'Unsupported county geometry {geom.geom_type}')


def bundle_from_data(data: pd.DataFrame, metrics: dict, counties):
    cells = sorted(data.grid_cell.unique(), key=lambda v: tuple(map(int, v.split('_'))))
    cell_index = {cell: ix for ix, cell in enumerate(cells)}
    dates = sorted(data.target_date_local.unique().tolist())
    cell_list = [{'id': cell, 'ix': int(cell.split('_')[0]), 'iy': int(cell.split('_')[1])} for cell in cells]
    all_days = {}
    for day, frame in data.groupby('target_date_local', sort=True):
        probs = [None] * len(cells)
        fire_counts = [0] * len(cells)
        for row in frame.itertuples(index=False):
            idx = cell_index[row.grid_cell]
            p = row.model_probability
            probs[idx] = round(float(p), 8) if pd.notna(p) else None
            fire_counts[idx] = int(row.fire_count)
        if len(frame) != len(cells):
            raise ValueError(f'{day}: not all cells are present; missing cells must be explicit')
        all_days[day] = {'probability': probs, 'fire_count': fire_counts}
    to_geo = Transformer.from_crs(CRS, 'EPSG:4326', always_xy=True)
    shapes = []
    for item in cell_list:
        ix, iy = item['ix'], item['iy']
        coords = [(ix * CELL_SIZE, iy * CELL_SIZE), ((ix+1)*CELL_SIZE, iy*CELL_SIZE),
                  ((ix+1)*CELL_SIZE, (iy+1)*CELL_SIZE), (ix*CELL_SIZE, (iy+1)*CELL_SIZE),
                  (ix*CELL_SIZE, iy*CELL_SIZE)]
        shapes.append({'type': 'Feature', 'id': item['id'],
                       'properties': {'grid_cell': item['id']},
                       'geometry': {'type': 'Polygon', 'coordinates': [
                           [[round(float(x), 7), round(float(y), 7)]
                            for x, y in (to_geo.transform(a, b) for a, b in coords)]]}})
    counties_ll = counties.to_crs('EPSG:4326')
    county_features = []
    for _, row in counties_ll.iterrows():
        county_features.append({'type': 'Feature', 'properties':
                                {'GEOID': row.GEOID, 'name': COUNTIES[row.GEOID]},
                                'geometry': row.geometry.__geo_interface__})
    return {'schema_version': 'emberscope-ai-v1',
            'note': 'Retrospective research scores, not as-issued forecasts, danger ratings or calibrated probabilities',
            'model': {'trained_years': metrics.get('training', {}).get('years', [2018]),
                      'validation_years': metrics.get('validation', {}).get('years', [2019, 2020, 2021]),
                      'validation': metrics['validation']},
            'grid': {'crs': CRS, 'cell_size_m': CELL_SIZE, 'cells': cell_list,
                     'geojson': {'type': 'FeatureCollection', 'features': shapes}},
            'county_boundaries': {'type': 'FeatureCollection', 'features': county_features},
            'dates': all_days}


def draw_preview(bundle, counties, output_path: Path):
    cells = bundle['grid']['cells']
    indices_x = sorted({v['ix'] for v in cells})
    indices_y = sorted({v['iy'] for v in cells})
    by_x = {value: i for i, value in enumerate(indices_x)}
    by_y = {value: i for i, value in enumerate(indices_y)}
    coords = [(by_y[v['iy']], by_x[v['ix']]) for v in cells]
    xs = [(ix + .5) * CELL_SIZE for ix in indices_x]
    ys = [(iy + .5) * CELL_SIZE for iy in indices_y]
    dates = list(bundle['dates'])
    shape = (len(indices_y), len(indices_x))
    mask = np.full(shape, np.nan)
    for r, c in coords:
        mask[r, c] = 1
    daily = {}
    all_values = []
    for day, record in bundle['dates'].items():
        risk = np.full(shape, np.nan)
        custom = np.full(shape, '', dtype=object)
        fire_x, fire_y, fire_t = [], [], []
        for (r, c), cell, prob, count in zip(coords, cells, record['probability'], record['fire_count']):
            custom[r, c] = f"{cell['id']} · {count} reported fires"
            if prob is not None:
                risk[r, c] = prob * 100
                all_values.append(prob * 100)
            if count > 0:
                fire_x.append((cell['ix']+.5)*CELL_SIZE)
                fire_y.append((cell['iy']+.5)*CELL_SIZE)
                fire_t.append(f"{cell['id']}: {count} reported wildfire discoveries")
        daily[day] = {'z': np.where(np.isfinite(risk), np.round(risk, 5), None).tolist(),
                      'custom': custom.tolist(), 'fx': fire_x, 'fy': fire_y,
                      'ft': fire_t, 'scored': sum(v is not None for v in record['probability']),
                      'fires': sum(v > 0 for v in record['fire_count'])}
    max_color = max(.5, float(np.percentile(all_values, 99.5))) if all_values else 1.0
    first = daily[dates[0]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=mask, colorscale=[[0,'#4c5b6d'],[1,'#4c5b6d']],
                              zmin=0, zmax=1, showscale=False, hoverinfo='skip', name='Not scored'))
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=first['z'], zmin=0, zmax=max_color,
                              colorscale=[[0,'#41358a'],[.22,'#3375eb'],[.45,'#23c4c8'],
                                          [.67,'#8ce943'],[.84,'#ffc341'],[1,'#e85c36']],
                              colorbar={'title': 'Model %', 'ticksuffix':'%'},
                              customdata=first['custom'],
                              hovertemplate='%{customdata}<br>Experimental score: %{z:.3f}%<extra></extra>',
                              name='Model estimate'))
    line_x, line_y = [], []
    for _, row in counties.iterrows():
        for ring in geometry_rings(row.geometry):
            xp, yp = ring.xy
            line_x.extend(list(xp)+[None])
            line_y.extend(list(yp)+[None])
    fig.add_trace(go.Scattergl(x=line_x, y=line_y, mode='lines',
                               line={'color':'#e5f0fe','width':2}, name='County boundaries', hoverinfo='skip'))
    points = counties.geometry.representative_point()
    fig.add_trace(go.Scattergl(x=points.x, y=points.y, mode='text',
                               text=[COUNTIES[fips] for fips in counties.GEOID],
                               textfont={'color':'#eef3fc','size':11},
                               textposition='middle center', name='County names', hoverinfo='skip'))
    fig.add_trace(go.Scattergl(x=first['fx'], y=first['fy'], text=first['ft'],
                               mode='markers', marker={'size':10,'color':'#ffc760',
                                                       'line':{'color':'#172435','width':1}},
                               hovertemplate='%{text}<extra></extra>', name='Reported fires'))
    fig.update_layout(template='plotly_dark', height=670, margin={'l':25,'r':15,'t':20,'b':15},
                      paper_bgcolor='#111b2a',plot_bgcolor='#111b2a',
                      xaxis={'visible':False},yaxis={'visible':False,'scaleanchor':'x','scaleratio':1},
                      legend={'orientation':'h','y':-0.02},hovermode='closest')
    plot_html = pio.to_html(fig, include_plotlyjs='inline', full_html=False,
                            div_id='ai-map',config={'responsive':True,'displaylogo':False})
    v = bundle['model']['validation']
    cards = [('Trained / tested', '2018 / 2019–21'),
             ('Historical rows evaluated', f"{v['scored_rows']:,}"),
             ('Days with reported fires', f"{v['positive_rate']*100:.3f}%"),
             ('Average precision', f"{v['average_precision']*100:.2f}%"),
             ('ROC AUC', f"{v['roc_auc']:.3f}"),
             ('Top 1% recall', f"{v['top_1pct']['recall']*100:.2f}%")]
    card_html = ''.join(f'<div class="card"><small>{name}</small><strong>{value}</strong></div>' for name,value in cards)
    choices = ''.join(f'<option value="{day}">{day}</option>' for day in dates)
    # Replace angle brackets so this JSON cannot prematurely close the inline script.
    js_data = json.dumps(daily, separators=(',', ':'), allow_nan=False).replace('<','\\u003c')
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EmberScope · Historical AI Explorer</title><style>
*{box-sizing:border-box}body{margin:0;background:#0a1421;color:#eef3fa;font:14px system-ui,Segoe UI,sans-serif}
main{max-width:1350px;margin:auto;padding:22px}.eyebrow{color:#ffb06b;letter-spacing:1.4px;font-size:11px;font-weight:700}
h1{margin:5px 0 6px;font-size:28px}p{color:#a8b9cc;line-height:1.55}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:19px 0}
.card,.panel{background:#132235;border:1px solid #274055;border-radius:11px;padding:15px}.card small{display:block;color:#adc1d1;margin-bottom:8px}.card strong{font-size:22px}
.header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}select{color:#eef3fa;background:#203850;padding:9px;border:1px solid #45657f;border-radius:7px}
.guide{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:16px 0 18px}.guide h2{font-size:16px;margin:0 0 7px}.guide p{margin:7px 0;font-size:13px}
.guide ul{padding-left:19px;margin:9px 0 2px;color:#cbd9e7;line-height:1.55}.guide li{margin:5px 0}.guide .note{font-size:12px;color:#a8b9cc}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:7px;vertical-align:-1px}.warm{background:linear-gradient(90deg,#41358a,#23c4c8,#ffc341,#e85c36);width:38px}.missing{background:#4c5b6d}.observed{background:#ffc760;border-radius:50%}.boundaries{background:#e5f0fe;height:3px;vertical-align:3px}
.mini{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:8px 0 4px}.mini span{font-size:12px;color:#c9d8e5}.readout{font-size:12px;color:#b9cad9;margin:8px 0 5px}
details{margin-top:10px;border-top:1px solid #355066;padding-top:10px;color:#b9cad9;font-size:12px;line-height:1.55}summary{cursor:pointer;color:#e3eef9;font-weight:650}
footer{margin:13px 0;color:#a4b4c5;font-size:12px}@media(max-width:780px){.guide{grid-template-columns:1fr}}@media(max-width:700px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><main><div class="eyebrow">EMBERSCOPE / AI RESEARCH</div>
<h1>Historical wildfire occurrence — model explorer</h1>
<p>Non-LA cells · USDA reported fire discoveries + gridMET historical reconstruction · Experiment only, not real-time or calibrated warning probabilities.</p>
<div class="cards">__CARDS__</div>
<section class="guide" aria-label="Understanding the model and map">
<div class="panel"><h2>What is this AI model?</h2>
<p>A <b>logistic regression</b> baseline that looks for patterns linking earlier weather and season to whether a wildfire discovery was reported in a 10 km cell on a future target date.</p>
<p><b>Inputs:</b> temperature, humidity, wind, daily rainfall, previous seven-day rainfall, and time of year. Its weather measurements come from <b>two calendar days before</b> the selected target date.</p>
<p><b>Training:</b> 2018 outside Los Angeles. <b>Testing:</b> 2019–2021 outside Los Angeles. Los Angeles has been held out for a separate evaluation.</p>
<p class="note"><b>Research only.</b> These are retrospective, uncalibrated model scores—not operational forecasts, NFDRS4 fire-danger ratings, or confirmed real-world ignition probabilities.</p>
<details><summary>What do the performance numbers mean?</summary>
<p><b>Average precision:</b> how effectively the model ranks reported-fire days near the top; a random ranking would score about __BASE_RATE__% on this test set. <b>ROC AUC:</b> 0.5 is no better than random ordering; 1.0 is perfect. <b>Top 1% recall:</b> the share of all fire-positive cell-days found among the highest-scored 1%.</p>
</details></div>
<div class="panel"><h2>How do I read the map?</h2>
<ul>
<li><span class="swatch warm" aria-hidden="true"></span><b>Colored squares:</b> one 10 km cell each. Cooler colors = lower model scores; warmer colors = higher scores <em>on this map's scale</em>, not official danger levels.</li>
<li><span class="swatch observed" aria-hidden="true"></span><b>Yellow dots:</b> locations with at least one wildfire discovery reported in USDA FPA-FOD for that selected day.</li>
<li><span class="swatch missing" aria-hidden="true"></span><b>Gray squares:</b> no score because the historical weather data was insufficient. They do <b>not</b> mean zero fire risk.</li>
<li><span class="swatch boundaries" aria-hidden="true"></span><b>Light county outlines and names:</b> geographic reference from the U.S. Census; Los Angeles has no model scores here.</li>
</ul>
<p class="note"><b>Read the percentage carefully:</b> 0.30% means 0.003 as a model score, <b>not 30%</b>. Color intensity compares experimental scores; the model has not been calibrated for public warnings. Hover over a cell for its exact value.</p>
</div></section>
<section class="panel"><div class="header"><div><b>Historical AI occurrence scores · 10 km grid</b>
<p>Pick a target date to compare the model's estimate with reported wildfire discoveries.</p></div>
<label>Target date <select id="map-date">__CHOICES__</select></label></div><div id="day-status" class="readout" aria-live="polite"></div>__PLOT__</section>
<footer>Source: USDA FPA-FOD fire discovery labels, gridMET reconstructed weather, U.S. Census 2024 boundaries. A day with no reported fire is not proof that no fire occurred. Non-LA research baseline only.</footer>
</main><script>
const daily=__DAILY__;
const sel=document.getElementById('map-date');
function setDay(day){const v=daily[day];Plotly.restyle('ai-map',{z:[v.z],customdata:[v.custom]},[1]);
Plotly.restyle('ai-map',{x:[v.fx],y:[v.fy],text:[v.ft]},[4]);
const weatherDate=new Date(Date.parse(day+'T00:00:00Z')-2*86400000).toISOString().slice(0,10);
document.getElementById('day-status').textContent=`Target: ${day} · Weather input date: ${weatherDate} · ${v.scored.toLocaleString()} scored cells · ${v.fires} cells with a reported discovery`;}
sel.addEventListener('change',()=>setDay(sel.value));setDay(sel.value);
</script></body></html>'''
    html = (html.replace('__CARDS__',card_html)
            .replace('__BASE_RATE__', f"{v['positive_rate']*100:.3f}")
            .replace('__CHOICES__',choices)
            .replace('__PLOT__',plot_html)
            .replace('__DAILY__',js_data))
    output_path.write_text(html,encoding='utf-8')


def add_ai_tab(main_dashboard: Path, ai_preview: Path):
    """Create an opt-in AI tab on a COPY of a generated static dashboard."""
    source = main_dashboard.read_text(encoding='utf-8')
    if '</body>' not in source.lower():
        raise ValueError('Expected an HTML file with a closing body tag')
    if 'id="es-ai-open"' in source:
        raise ValueError('This dashboard already has an AI integration control')
    target = main_dashboard.with_name(main_dashboard.stem + '_with_ai.html')
    relative = os.path.relpath(ai_preview.resolve(), target.parent.resolve()).replace('\\', '/')
    path = escape(relative, quote=True)
    injection = f'''<style id="es-ai-bridge-style">
#es-ai-open{{position:fixed;right:18px;bottom:18px;z-index:2147483000;padding:14px 19px;
background:#ed9854;color:#101d2a;border:0;border-radius:24px;font:700 14px system-ui;cursor:pointer;
box-shadow:0 8px 25px #0006}}
#es-ai-panel[hidden]{{display:none!important}}
#es-ai-panel{{position:fixed;inset:18px;z-index:2147483001;background:#101d2d;border:1px solid #385066;
border-radius:13px;overflow:hidden;box-shadow:0 0 0 200vmax #000a;display:flex;flex-direction:column}}
#es-ai-head{{display:flex;align-items:center;gap:14px;justify-content:space-between;padding:10px 15px;
color:#e4f1fb;background:#1e3348;font:14px system-ui}}
#es-ai-head button{{border:1px solid #8a9dad;border-radius:6px;background:#324e64;color:white;
padding:8px 12px;cursor:pointer}}#es-ai-panel iframe{{border:0;flex:1;width:100%;background:#0a1421}}
@media(max-width:700px){{#es-ai-panel{{inset:4px}}}}
</style>
<button type="button" id="es-ai-open" aria-expanded="false" aria-controls="es-ai-panel">AI research explorer</button>
<div id="es-ai-panel" role="dialog" aria-label="Historical AI research explorer" hidden>
<div id="es-ai-head"><strong>Historical AI occurrence model · 2018 trained / 2019–21 validated · NOT LIVE</strong>
<button type="button" id="es-ai-close">Close AI explorer</button></div>
<iframe title="Historical AI model and Census boundaries" src="{path}" loading="lazy"></iframe></div>
<script id="es-ai-bridge-script">
(()=>{{'use strict';const open=document.getElementById('es-ai-open');
const panel=document.getElementById('es-ai-panel');const close=document.getElementById('es-ai-close');
function toggle(show){{panel.hidden=!show;open.setAttribute('aria-expanded',String(show));
if(show)close.focus();else open.focus();}}
open.addEventListener('click',()=>toggle(true));close.addEventListener('click',()=>toggle(false));
document.addEventListener('keydown',e=>{{if(e.key==='Escape'&&!panel.hidden)toggle(false);}});
}})();</script>
'''
    pos = source.lower().rfind('</body>')
    target.write_text(source[:pos] + injection + source[pos:], encoding='utf-8')
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', type=Path, default=PREDICTIONS)
    ap.add_argument('--metrics', type=Path, default=METRICS)
    ap.add_argument('--boundaries', type=Path, default=BOUNDARIES)
    ap.add_argument('--output-dir', type=Path, default=OUT)
    ap.add_argument('--embed-into', type=Path, default=None,
                    help='Optional generated EmberScope HTML; writes an *_with_ai.html copy')
    args = ap.parse_args(argv)
    df, metrics = checked_input(args.predictions, args.metrics)
    county_shapes = load_counties(args.boundaries)
    bundle = bundle_from_data(df, metrics, county_shapes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / 'ai_map_bundle.json'
    html_path = args.output_dir / 'ai_map_preview.html'
    json_path.write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    draw_preview(bundle,county_shapes,html_path)
    print(f'Exported {len(bundle["dates"])} dates, {len(bundle["grid"]["cells"]):,} cells and {len(county_shapes)} county boundaries')
    print('Frontend bundle:',json_path)
    print('Offline AI preview:',html_path)
    if args.embed_into is not None:
        integrated = add_ai_tab(args.embed_into, html_path)
        print('Integrated EmberScope preview:', integrated)
    print('No retraining or weather downloads performed.')


if __name__ == '__main__':
    main()
