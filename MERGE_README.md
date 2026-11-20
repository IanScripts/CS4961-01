# CS4961-01 — Wildfire Prediction

## Requirements

- Python 3
- Required Python packages (see `requirements.txt`)

## Setup

```bash
git clone https://github.com/IanScripts/CS4961-01.git
cd CS4961-01
pip install -r requirements.txt
```

## Running the Project

**1. Collect weather data**

```cmd
python src/weather_apis/open_meteo_api.py
```

**2. Generate experimental wildfire scores**

```cmd
python scripts/score_open_meteo_csv.py
```

**3. Open the EmberScope dashboard (Windows)**

```cmd
start "" "dashboard.html"
```

**4. Run tests (optional)**

```cmd
python -m unittest discover -s tests -p "test*.py"
```

## Notes

- Dashboard HTML files must already exist or be generated before opening.
- The AI research dashboard requires the saved prediction CSV, metrics JSON, and Census county boundaries.
```cmd
python scripts/train_weather_baseline.py
```
```cmd
mkdir wildfire-datasets\boundaries
curl -L "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_500k.zip" -o "wildfire-datasets\boundaries\counties_2024.zip"
```
```cmd
python -m scripts.export_ai_dashboard
```
```cmd
start "" "data\ai\weather_baseline_v1\ai_map_preview.html"
```
- The wildfire scores are experimental and should not be treated as official fire warnings.
