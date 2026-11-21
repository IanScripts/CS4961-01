"""Train the first SoCal weather/season wildfire occurrence baseline.

Run from the CS4961-01 repository root:
    python -m scripts.train_weather_baseline
    start "" "data\\ai\\weather_baseline_v1\\wildfire_model_dashboard.html"

Uses existing monthly data: all 2018 non-LA cell-days for training,
2019-2021 non-LA cell-days for forward validation. LA is never used.
Only historical/reconstructed inputs: NOT an operational fire forecast.
"""
from __future__ import annotations

import argparse
import json
import math
from calendar import monthrange
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Deliberately fixed, documented evaluation split. Do not tune on Los Angeles.
TRAIN_YEAR = 2018
VALIDATION_YEARS = (2019, 2020, 2021)
MAP_MONTH = "2021-12"
WEATHER_FEATURES = (
    "tmax_k", "rh_min_pct", "wind_mean_ms", "precip_mm",
    "precip_previous_7d_mm",
)
MODEL_FEATURES = (*WEATHER_FEATURES, "season_sin", "season_cos")
READ_COLUMNS = [
    "grid_cell", "target_date_local", "weather_date_gridmet", *WEATHER_FEATURES,
    "fire_count", "fire_label", "coverage_min_fraction", "weather_lag_days",
    "spatial_method",
]
FLOAT_TYPES = {name: "float32" for name in WEATHER_FEATURES}
FLOAT_TYPES["coverage_min_fraction"] = "float32"


def months_in_year(year):
    return [f"{year}-{m:02d}" for m in range(1, 13)]


def load_month(data_root: Path, month: str, expected_config_hash: str | None):
    """Load one *full-region* month with lightweight correctness checks."""
    csv_path = data_root / f"regional_weather_{month.replace('-', '_')}" / "training_features.csv"
    manifest_path = csv_path.with_suffix(".manifest.json")
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing completed month {month}: {csv_path}\n"
            "Build that month with scripts.build_regional_weather first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != f"regional_gridmet_fire_labels_{month}":
        raise ValueError(f"Wrong dataset manifest for {month}")
    if manifest.get("sampling") != "entire eligible non-LA population":
        raise ValueError(f"{month} is not a full non-LA regional dataset")
    source_hash = manifest.get("config_sha256")
    if not source_hash or (expected_config_hash is not None and source_hash != expected_config_hash):
        raise ValueError(f"{month} uses a different geographic config; don't mix populations")
    if manifest.get("weather_lag_days") != 2:
        raise ValueError(f"{month}: weather lag changed")

    df = pd.read_csv(
        csv_path,
        usecols=READ_COLUMNS,
        dtype={**FLOAT_TYPES, "fire_count": "int32", "fire_label": "int8"},
        parse_dates=["target_date_local", "weather_date_gridmet"],
    )
    year, month_num = map(int, month.split("-"))
    expected_rows = int(manifest["cells"]) * monthrange(year, month_num)[1]
    if len(df) != expected_rows or len(df) != int(manifest["rows"]):
        raise ValueError(f"{month}: expected {expected_rows} rows, found {len(df)}")
    if df.duplicated(["grid_cell", "target_date_local"]).any():
        raise ValueError(f"{month}: duplicate cell-date keys")
    if not df["target_date_local"].dt.strftime("%Y-%m").eq(month).all():
        raise ValueError(f"{month}: target dates outside the month")
    if not (df["target_date_local"] - df["weather_date_gridmet"]).dt.days.eq(2).all():
        raise ValueError(f"{month}: invalid historical weather-date alignment")
    if not df["fire_label"].isin([0, 1]).all():
        raise ValueError(f"{month}: nonbinary wildfire label")
    if not (df["fire_label"] == df["fire_count"].gt(0).astype("int8")).all():
        raise ValueError(f"{month}: wildfire label and count disagree")
    if not df["spatial_method"].eq("area_weighted_gridmet_pixels").all():
        raise ValueError(f"{month}: unknown spatial method")
    if int(df["fire_label"].sum()) != int(manifest["positive_cell_days"]):
        raise ValueError(f"{month}: fire-positive count differs from manifest")
    return df, source_hash


def add_features(frame: pd.DataFrame):
    """Use only earlier weather and calendar features known by the target date."""
    day = frame["target_date_local"].dt.dayofyear.to_numpy()
    frame = frame.copy()
    frame["season_sin"] = np.sin(2.0 * np.pi * day / 365.25).astype("float32")
    frame["season_cos"] = np.cos(2.0 * np.pi * day / 365.25).astype("float32")
    return frame


def completed(frame):
    return frame.loc[frame[list(WEATHER_FEATURES)].notna().all(axis=1)]


def month_iterator(years):
    for year in years:
        for month in months_in_year(year):
            yield month


def safe_metric(ys, p):
    if len(np.unique(ys)) < 2:
        return None, None
    return float(average_precision_score(ys, p)), float(roc_auc_score(ys, p))


def top_budget(y, p, fraction):
    """An *evaluation* summary, not an operational alert threshold."""
    n = max(1, int(math.ceil(len(p) * fraction)))
    top = np.argpartition(p, -n)[-n:]
    hits = int(np.sum(y[top]))
    positives = int(np.sum(y))
    return {
        "fraction": float(fraction), "flagged_rows": n, "hits": hits,
        "precision": hits / n,
        "recall": hits / positives if positives else None,
    }


def cell_polygon_geojson(cells):
    """Four projected corners, converted to lon/lat for map display only."""
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    features = []
    centers = {}
    for cell in cells:
        ix, iy = (int(part) for part in cell.split("_"))
        x0, x1 = ix * 10_000, (ix + 1) * 10_000
        y0, y1 = iy * 10_000, (iy + 1) * 10_000
        coords = [transformer.transform(x, y) for x, y in (
            (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)
        )]
        center_lon, center_lat = transformer.transform((x0 + x1) / 2, (y0 + y1) / 2)
        centers[cell] = (center_lat, center_lon)
        features.append({
            "type": "Feature", "id": cell, "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[list(pair) for pair in coords]]},
        })
    return {"type": "FeatureCollection", "features": features}, centers


def map_fig_and_dates(frame: pd.DataFrame):
    """Offline 10-km cell map using a regular projected grid, no tile server.

    The original MapLibre version can render blank in browsers when WebGL or
    map tiles are unavailable. EPSG:5070 grid coordinates are already a valid
    spatial representation, so Plotly heatmap + scatter avoids that dependency.
    """
    all_cells = sorted(frame["grid_cell"].unique().tolist())
    cell_coords = {cell: tuple(map(int, cell.split("_"))) for cell in all_cells}
    ix_values = sorted({p[0] for p in cell_coords.values()})
    iy_values = sorted({p[1] for p in cell_coords.values()})
    col_lookup = {ix: i for i, ix in enumerate(ix_values)}
    row_lookup = {iy: i for i, iy in enumerate(iy_values)}
    xs = [(ix + 0.5) * 10000 for ix in ix_values]
    ys = [(iy + 0.5) * 10000 for iy in iy_values]
    height, width = len(ys), len(xs)

    base = [[None] * width for _ in range(height)]
    positions = {}
    for cell, (ix, iy) in cell_coords.items():
        row, col = row_lookup[iy], col_lookup[ix]
        positions[cell] = (row, col)
        base[row][col] = 1

    dates = sorted(frame["target_date_local"].dt.strftime("%Y-%m-%d").unique().tolist())
    rows_by_date = {
        day: part.set_index("grid_cell")
        for day, part in frame.groupby(frame["target_date_local"].dt.strftime("%Y-%m-%d"), sort=True)
    }
    map_data = {}
    for day in dates:
        day_rows = rows_by_date[day].reindex(all_cells)
        risks = [[None] * width for _ in range(height)]
        custom = [[["", 0, None] for _ in range(width)] for _ in range(height)]
        fires_x, fires_y, fires_text = [], [], []
        for cell in all_cells:
            row, col = positions[cell]
            record = day_rows.loc[cell]
            prediction = float(record["model_probability"])
            tmax = float(record["tmax_k"])
            fire_count = int(record["fire_count"]) if pd.notna(record["fire_count"]) else 0
            risks[row][col] = round(prediction * 100, 5) if np.isfinite(prediction) else None
            custom[row][col] = [cell, fire_count,
                                round(tmax - 273.15, 1) if np.isfinite(tmax) else None]
            if fire_count > 0:
                ix, iy = cell_coords[cell]
                fires_x.append((ix + 0.5) * 10000)
                fires_y.append((iy + 0.5) * 10000)
                fires_text.append(f"{cell}: {fire_count} reported discoveries")
        map_data[day] = {
            "risk": risks,
            "custom": custom,
            "fire_x": fires_x,
            "fire_y": fires_y,
            "fire_text": fires_text,
            "positive_cells": len(fires_text),
            "scored_cells": int(day_rows["model_probability"].notna().sum()),
        }

    flat_scores = [v for info in map_data.values() for line in info["risk"]
                   for v in line if v is not None]
    scale_max = max(0.5, float(np.percentile(flat_scores, 99.5)) if flat_scores else 1.0)
    initial = map_data[dates[0]]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=base, zmin=0, zmax=1,
        colorscale=[[0, "#465467"], [1, "#465467"]],
        showscale=False, hoverinfo="skip", xgap=1, ygap=1,
        name="Study footprint",
    ))
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=initial["risk"], customdata=initial["custom"],
        zmin=0, zmax=scale_max, colorscale="Turbo", xgap=1, ygap=1,
        colorbar={"title": {"text": "Model %"}, "ticksuffix": "%", "thickness": 15,
                  "tickfont": {"color": "#dde7f0"}},
        hovertemplate=("<b>Cell %{customdata[0]}</b><br>Model estimate: %{z:.3f}%"
                       "<br>Reported fires: %{customdata[1]}"
                       "<br>Tmax (prior weather day): %{customdata[2]}°C<extra></extra>"),
        name="Weather + season model",
    ))
    fig.add_trace(go.Scatter(
        x=initial["fire_x"], y=initial["fire_y"], text=initial["fire_text"],
        mode="markers", marker={"size": 9, "color": "#ffdd69", "opacity": 0.95,
                                 "line": {"color": "#121b26", "width": 1}},
        hovertemplate="<b>%{text}</b><extra>Reported fire</extra>",
        name="Actual fire-positive cells",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#101924", plot_bgcolor="#101924",
        margin={"l": 8, "r": 8, "t": 8, "b": 8}, height=580,
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1,
               "range": [min(xs) - 10000, max(xs) + 10000]},
        yaxis={"visible": False, "range": [min(ys) - 10000, max(ys) + 10000]},
        legend={"orientation": "h", "x": 0.02, "y": 0.02,
                "bgcolor": "rgba(8,15,25,0.8)"},
    )
    return fig, map_data, dates, scale_max


def plot_precision_recall(y, probs):
    precision, recall, _ = precision_recall_curve(y, probs)
    stride = max(1, len(precision) // 750)
    idx = np.arange(0, len(precision), stride)
    if idx[-1] != len(precision) - 1:
        idx = np.append(idx, len(precision) - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall[idx], y=precision[idx], name="Weather + season", mode="lines",
                             line={"color": "#ff9d5c", "width": 2.5}))
    prior = float(np.mean(y))
    fig.add_hline(y=prior, line_color="#6ed3cb", line_dash="dash",
                  annotation_text="Random ranking / prevalence", annotation_font_color="#6ed3cb")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, max(.05, float(np.max(precision)) * 1.03)],
                      xaxis_range=[0, 1], showlegend=False)
    return fig


def plot_monthly_trend(records):
    df = pd.DataFrame(records)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["observed_rate"] * 100,
                             name="Observed reported-fire rate", mode="lines+markers",
                             line={"color": "#ffb16e", "width": 2.5}))
    fig.add_trace(go.Scatter(x=df["month"], y=df["predicted_mean"] * 100,
                             name="Mean model estimate", mode="lines+markers",
                             line={"color": "#78d4cb", "width": 2.5}))
    fig.update_layout(xaxis_title="Validation month", yaxis_title="Positive cell-days (%)",
                      hovermode="x unified", legend={"orientation": "h", "y": 1.18})
    return fig


def plot_calibration(y, p):
    # Equal-population score bins. Descriptive, NOT a calibration procedure.
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        edges = np.linspace(float(p.min()) - 1e-8, float(p.max()) + 1e-8, 11)
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    pred = []
    actual = []
    for index in range(len(edges) - 1):
        mask = bins == index
        if mask.any():
            pred.append(float(p[mask].mean()) * 100)
            actual.append(float(y[mask].mean()) * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred, y=actual, mode="lines+markers",
                             name="Observed vs estimated", line={"color": "#ffb16e", "width": 2.5}))
    bound = max(pred + actual + [.25]) * 1.07
    fig.add_trace(go.Scatter(x=[0, bound], y=[0, bound], mode="lines", name="Ideal agreement",
                             line={"color": "#78d4cb", "dash": "dash"}))
    fig.update_layout(xaxis_title="Mean model estimate (%)", yaxis_title="Observed positive rate (%)")
    return fig


def make_dashboard(output: Path, map_frame: pd.DataFrame, metrics: dict,
                   y_valid: np.ndarray, p_valid: np.ndarray, monthly_records: list,
                   coefficients: np.ndarray):
    map_fig, dates_data, dates, scale_max = map_fig_and_dates(map_frame)
    pr = plot_precision_recall(y_valid, p_valid)
    trend = plot_monthly_trend(monthly_records)
    calibration = plot_calibration(y_valid, p_valid)
    coefficients_fig = go.Figure(go.Bar(
        x=list(coefficients), y=list(MODEL_FEATURES), orientation="h",
        marker_color=["#ff9d5c" if c > 0 else "#78d4cb" for c in coefficients],
        hovertemplate="%{y}: %{x:.3f}<extra>Standardized coefficient</extra>",
    ))
    coefficients_fig.update_layout(xaxis_title="Logistic regression weight (standardized features)",
                                   yaxis={"autorange": "reversed"})
    for chart in (pr, trend, calibration, coefficients_fig):
        chart.update_layout(template="plotly_dark", paper_bgcolor="#132131", plot_bgcolor="#132131",
                            font={"color": "#d8e5f0"}, margin={"t": 24, "b": 52, "l": 65, "r": 20}, height=300)
    map_div = pio.to_html(map_fig, include_plotlyjs="inline", full_html=False,
                          div_id="risk-map", config={"responsive": True, "displaylogo": False})
    plots = [pio.to_html(fig, include_plotlyjs=False, full_html=False,
                         config={"responsive": True, "displaylogo": False})
             for fig in (pr, trend, calibration, coefficients_fig)]
    cards = [
        ("Validation observations", f"{metrics['validation']['scored_rows']:,}", "2019–2021, weather-complete"),
        ("Fire-positive rate", f"{metrics['validation']['positive_rate'] * 100:.3f}%", "Reported discoveries in FPA-FOD"),
        ("Average precision", f"{metrics['validation']['average_precision']:.4f}", "Higher is better; compare to prevalence"),
        ("Random-ranking baseline AP", f"{metrics['validation']['positive_rate']:.4f}", "Same validation prevalence"),
        ("ROC AUC", f"{metrics['validation']['roc_auc']:.3f}", "0.5 ≈ random ranking"),
        ("Top 1% captures", f"{metrics['validation']['top_1pct']['recall'] * 100:.1f}%", "Of reported positive cell-days"),
    ]
    cards_html = "".join(
        f'<div class="stat"><div class="k">{escape(name)}</div>'
        f'<div class="v">{escape(value)}</div><div class="sub">{escape(note)}</div></div>'
        for name, value, note in cards
    )
    choices = "".join(f'<option value="{date}">{date}</option>' for date in dates)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EmberScope Research • Weather-Based Wildfire Baseline</title>
<style>
:root{{--bg:#0b1420;--surface:#132131;--edge:#23364a;--fg:#edf3f9;--muted:#9eb1c4;--accent:#ff9d5c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1450px;padding:32px 28px 65px;margin:auto}}.eyebrow{{color:var(--accent);font-weight:750;font-size:12px;letter-spacing:2px;text-transform:uppercase}}
h1{{font-size:clamp(27px,4vw,46px);letter-spacing:-1px;margin:7px 0 10px}}h2{{font-size:18px;margin:0 0 8px}}p{{color:var(--muted);line-height:1.6}}
header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:24px}}
.tag{{background:#1d3641;color:#a0f2df;border:1px solid #2f565a;padding:9px 12px;border-radius:99px;font-size:12px;font-weight:700;white-space:nowrap}}
.stats{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0}}
.stat{{background:var(--surface);border:1px solid var(--edge);padding:18px;border-radius:13px}}
.k{{color:var(--muted);font-size:12px}}.v{{font-size:28px;font-weight:750;letter-spacing:-.7px;margin-top:6px}}.sub{{font-size:11px;color:var(--muted);margin-top:5px}}
.panel{{background:var(--surface);border:1px solid var(--edge);border-radius:16px;overflow:hidden;padding:20px;margin-bottom:14px}}
.panelhead{{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}select{{appearance:none;border:1px solid #456076;border-radius:9px;padding:10px 16px;background:#1a2b3e;color:#fff;font-size:15px;cursor:pointer}}
#risk-map{{min-height:580px}}.footnote{{font-size:12px;color:#95a9bb;line-height:1.6}}
.warning{{color:#e7c6a4;background:#30271d;padding:13px 17px;border:1px solid #604731;border-radius:9px;margin-top:22px}}
@media(max-width:850px){{.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid{{grid-template-columns:1fr}}header{{display:block}}.tag{{display:inline-block}}}}
</style></head><body><main>
<header><div><div class="eyebrow">EmberScope / Capstone research</div>
<h1>Wildfire Occurrence • AI Baseline</h1>
<p>Trained on historical 2018 non-LA cells; evaluated on unseen 2019–2021 non-LA dates.<br>
Historical discovery labels + area-weighted gridMET weather with a two-day calendar lag.</p></div>
<div class="tag">RESEARCH ONLY · NOT LIVE RISK</div></header>
<section class="stats">{cards_html}</section>
<section class="panel"><div class="panelhead"><div><h2>Model estimates across Southern California</h2>
<div class="footnote">Select a historical date. Colored squares show model estimates, yellow dots mark actual reported fire-positive cells; grey squares lack usable weather.</div></div>
<label>Target date &nbsp;<select id="date-select">{choices}</select></label></div>
<div id="date-summary" class="footnote"></div>{map_div}</section>
<section class="grid">
<div class="panel"><h2>Precision and recall</h2><div class="footnote">Does model ranking identify rare positive cells better than a random ranking?</div>{plots[0]}</div>
<div class="panel"><h2>Month-by-month: estimated vs observed</h2><div class="footnote">A basic check for changes across seasons and years.</div>{plots[1]}</div>
<div class="panel"><h2>Probability agreement</h2><div class="footnote">Descriptive score bins; these probabilities have NOT been independently calibrated.</div>{plots[2]}</div>
<div class="panel"><h2>What the model learned</h2><div class="footnote">Standardized model coefficients are associations, not causal explanations.</div>{plots[3]}</div>
</section>
<div class="warning"><b>Important:</b> This model estimates whether a wildfire discovery was <i>reported in the USDA archive</i>, not whether a new fire is certain to start. It is trained on retrospectively reconstructed weather, excludes 37 persistently incomplete coastal cells, and is not a real-time forecast. Los Angeles is still held out. The projected 10-km grid map and charts are fully embedded and work without map tiles.</div>
<p class="footnote">Evaluation uses natural cell-day prevalence (no resampling); the model was fit only on 2018. Dates on the map come from the 2021-12 validation month. Model estimates are not NFDRS4 fire-danger indices.</p>
</main><script>
const datesData={json.dumps(dates_data, separators=(',', ':'))};
const select=document.getElementById('date-select');
function setDay(date){{
  const d=datesData[date];
  Plotly.restyle('risk-map', {{z:[d.risk], customdata:[d.custom]}}, [1]);
  Plotly.restyle('risk-map', {{x:[d.fire_x], y:[d.fire_y], text:[d.fire_text]}}, [2]);
  document.getElementById('date-summary').textContent=`${{date}}  •  ${{d.scored_cells.toLocaleString()}} scored cells  •  ${{d.positive_cells}} cells with actual reported discoveries`;
}}
select.addEventListener('change',()=>setDay(select.value));setDay(select.value);
</script></body></html>'''
    output.write_text(html, encoding="utf-8")


def run(data_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Training period: 2018 only (non-LA); validation: 2019–2021 (non-LA)", flush=True)
    print("Loading 12 training months...", flush=True)
    train_parts = []
    hash_key = None
    train_total = 0
    for month in months_in_year(TRAIN_YEAR):
        frame, hash_key = load_month(data_root, month, hash_key)
        train_total += len(frame)
        train_parts.append(add_features(completed(frame))[[*MODEL_FEATURES, "fire_label"]])
    train_frame = pd.concat(train_parts, ignore_index=True)
    del train_parts
    x_train = train_frame[list(MODEL_FEATURES)].to_numpy(dtype=np.float64)
    y_train = train_frame["fire_label"].to_numpy(dtype=np.int8)
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training data must contain both positive and negative outcomes")
    print(f"Training: {len(y_train):,} complete rows, {int(y_train.sum()):,} positive", flush=True)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, solver="lbfgs"))
    model.fit(x_train, y_train)
    del x_train, train_frame
    joblib.dump({"model": model, "features": MODEL_FEATURES, "trained_year": 2018,
                 "weather_lag_days": 2, "config_sha256": hash_key},
                output_dir / "weather_baseline_2018.joblib")
    print("Model fitted. Evaluating 2019–2021 one month at a time...", flush=True)

    ys, ps = [], []
    validation_rows = scored_rows = 0
    monthly_records = []
    map_frame = None
    for month in month_iterator(VALIDATION_YEARS):
        frame, _ = load_month(data_root, month, hash_key)
        validation_rows += len(frame)
        frame = add_features(completed(frame))
        x = frame[list(MODEL_FEATURES)].to_numpy(dtype=np.float64)
        preds = model.predict_proba(x)[:, 1]
        label = frame["fire_label"].to_numpy(dtype=np.int8)
        ys.append(label)
        ps.append(preds)
        scored_rows += len(frame)
        monthly_records.append({
            "month": month, "rows": len(frame),
            "positives": int(label.sum()),
            "observed_rate": float(label.mean()),
            "predicted_mean": float(preds.mean()),
        })
        if month == MAP_MONTH:
            # Preserve missing-weather cells on the map as grey squares.
            original, _ = load_month(data_root, month, hash_key)
            frame_for_map = original.copy()
            frame_for_map["model_probability"] = np.nan
            frame_for_map.loc[frame.index, "model_probability"] = preds
            map_frame = frame_for_map
        print(f"  {month}: {len(label):,} scored, {int(label.sum()):,} positives", flush=True)
    if map_frame is None:
        raise ValueError(f"Map month {MAP_MONTH} not in validation period")

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    ap, roc = safe_metric(y, p)
    if ap is None:
        raise ValueError("Validation data has only one label class")
    top_one = top_budget(y, p, .01)
    top_five = top_budget(y, p, .05)
    metrics = {
        "purpose": "Retrospective research baseline; not an as-issued weather forecast",
        "model": "StandardScaler + LogisticRegression; weather and seasonal sin/cos",
        "features": list(MODEL_FEATURES),
        "training": {"years": [2018], "total_rows": train_total, "scored_rows": len(y_train),
                     "excluded_missing": train_total - len(y_train), "positive_rows": int(y_train.sum())},
        "validation": {
            "years": [2019, 2020, 2021], "total_rows": validation_rows,
            "scored_rows": scored_rows, "excluded_missing": validation_rows - scored_rows,
            "positive_rows": int(y.sum()), "positive_rate": float(y.mean()),
            "average_precision": ap, "roc_auc": roc,
            "brier_score": float(brier_score_loss(y, p)),
            "constant_training_prevalence_brier": float(brier_score_loss(y, np.full(len(y), y_train.mean()))),
            "top_1pct": top_one, "top_5pct": top_five,
        },
        "monthly": monthly_records,
        "map_month": MAP_MONTH, "manifest_config_sha256": hash_key,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "caveats": ["LA never used", "Drop rows missing weather", "FPA-FOD labels are reported discoveries",
                    "gridMET historical reconstructions are not real-time issued forecasts",
                    "Validation probabilities not independently calibrated"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    pred_file = output_dir / f"map_predictions_{MAP_MONTH.replace('-', '_')}.csv"
    map_frame[["grid_cell", "target_date_local", "weather_date_gridmet", "tmax_k",
               "rh_min_pct", "wind_mean_ms", "precip_mm", "precip_previous_7d_mm",
               "fire_count", "fire_label", "model_probability"]].to_csv(pred_file, index=False)
    dashboard = output_dir / "wildfire_model_dashboard.html"
    coeff = model.named_steps["logisticregression"].coef_.ravel()
    make_dashboard(dashboard, map_frame, metrics, y, p, monthly_records, coeff)
    print("\n--- VALIDATION RESULTS (2019–2021) ---")
    print(f"Scored rows: {scored_rows:,} / {validation_rows:,}")
    print(f"Reported-fire prevalence: {y.mean()*100:.3f}%")
    print(f"Average precision (AP): {ap:.5f} | random ranking baseline: {y.mean():.5f}")
    print(f"ROC AUC: {roc:.4f}")
    print(f"Top 1% alerts: precision {top_one['precision']*100:.2f}%, recall {top_one['recall']*100:.2f}%")
    print(f"Dashboard: {dashboard}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Map CSV: {pred_file}")
    print("RESEARCH ONLY: not a deployed or calibrated wildfire forecast.")


def repair_existing_map(output_dir: Path):
    """Replace only the map inside an existing HTML report, without retraining."""
    csv_path = output_dir / f"map_predictions_{MAP_MONTH.replace('-', '_')}.csv"
    original_path = output_dir / "wildfire_model_dashboard.html"
    result_path = output_dir / "wildfire_model_dashboard_offline_map.html"
    if not csv_path.is_file() or not original_path.is_file():
        raise FileNotFoundError("Missing previous map prediction CSV or original dashboard HTML")

    frame = pd.read_csv(csv_path, parse_dates=["target_date_local"])
    fig, dates_data, dates, _ = map_fig_and_dates(frame)
    map_div = pio.to_html(fig, include_plotlyjs="inline", full_html=False,
                          div_id="risk-map", config={"responsive": True, "displaylogo": False})
    page = original_path.read_text(encoding="utf-8")
    marker = '<div id="date-summary" class="footnote"></div>'
    begin = page.find(marker)
    if begin < 0:
        raise ValueError("Could not identify the map section in original dashboard HTML")
    end = page.find("</section>", begin)
    if end < 0:
        raise ValueError("Could not find map section closing tag")
    page = page[:begin] + marker + map_div + page[end:]

    script_marker = '<script>\nconst datesData='
    begin = page.rfind(script_marker)
    if begin < 0:
        raise ValueError("Could not identify original date-selector script")
    end = page.find("</script>", begin)
    if end < 0:
        raise ValueError("Could not find original date-selector closing tag")
    replacement = '''<script>
const datesData=__DATES_JSON__;
const select=document.getElementById('date-select');
function setDay(date){
  const d=datesData[date];
  Plotly.restyle('risk-map', {z:[d.risk], customdata:[d.custom]}, [1]);
  Plotly.restyle('risk-map', {x:[d.fire_x], y:[d.fire_y], text:[d.fire_text]}, [2]);
  document.getElementById('date-summary').textContent=`${date}  •  ${d.scored_cells.toLocaleString()} scored cells  •  ${d.positive_cells} cells with reported fires`;
}
select.addEventListener('change',()=>setDay(select.value));setDay(select.value);
</script>'''.replace("__DATES_JSON__", json.dumps(dates_data, separators=(",", ":")))
    page = page[:begin] + replacement + page[end + len("</script>"):]
    page = page.replace("Map basemap tiles require an internet connection; the HTML charts and data are embedded locally.",
                        "The projected grid map and all charts are embedded and work offline.")
    page = page.replace("Colored squares show model estimates, yellow dots mark actual reported fire-positive cells; grey squares lack usable weather.",
                        "Colored squares show model estimates; yellow dots show reported fires. Grey cells lack usable weather. This map uses a projected grid, not street-map tiles.")
    result_path.write_text(page, encoding="utf-8")
    print(f"Map repaired (no retraining): {result_path}")
    print(f"Days: {len(dates)} | cells: {frame['grid_cell'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ai"),
                        help="Directory containing regional_weather_YYYY_MM folders")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ai/weather_baseline_v1"))
    parser.add_argument("--repair-map", action="store_true",
                        help="Regenerate only the offline map from saved model outputs")
    args = parser.parse_args()
    if args.repair_map:
        repair_existing_map(args.output_dir)
    else:
        run(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
