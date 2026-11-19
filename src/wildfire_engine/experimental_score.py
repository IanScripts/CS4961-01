"""EmberScope experimental weather-driven fire-conduciveness index.

Not NFDRS, FWI, ignition probability, spread simulation, or official fire danger.
All thresholds and weights are illustrative and must be calibrated/validated before use.
"""
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from math import exp, sin, pi
from typing import Any

REGIONS = {
    "los-angeles": {"name": "Los Angeles Foothills", "county": "Los Angeles", "lat": 34.227, "lon": -118.103, "terrain": "Chaparral / foothills"},
    "san-diego": {"name": "San Diego Backcountry", "county": "San Diego", "lat": 32.925, "lon": -116.650, "terrain": "Scrub / backcountry"},
    "riverside": {"name": "Riverside / Inland Empire", "county": "Riverside", "lat": 33.927, "lon": -117.396, "terrain": "Inland valley"},
    "ventura": {"name": "Ventura County", "county": "Ventura", "lat": 34.370, "lon": -119.160, "terrain": "Coastal foothills"},
    "santa-barbara": {"name": "Santa Barbara Mountains", "county": "Santa Barbara", "lat": 34.470, "lon": -119.680, "terrain": "Coastal mountains"},
    "sacramento": {"name": "Sacramento Valley", "county": "Sacramento", "lat": 38.581, "lon": -121.494, "terrain": "Valley / grassland"},
    "redding": {"name": "Redding / Shasta", "county": "Shasta", "lat": 40.587, "lon": -122.392, "terrain": "Northern foothills"},
    "fresno": {"name": "Fresno / Central Valley", "county": "Fresno", "lat": 36.737, "lon": -119.787, "terrain": "Valley / grassland"},
    "santa-rosa": {"name": "Santa Rosa / Sonoma", "county": "Sonoma", "lat": 38.441, "lon": -122.714, "terrain": "North Bay hills"},
    "tahoe": {"name": "Lake Tahoe / Sierra", "county": "El Dorado", "lat": 38.945, "lon": -120.120, "terrain": "Montane forest"},
}

def clamp(v: float, a=0.0, b=1.0) -> float:
    return max(a, min(b, v))

def normalized(value: float, low: float, high: float) -> float:
    return clamp((value-low)/(high-low))

@dataclass(frozen=True)
class WeatherDay:
    day: str
    temperature_c: float
    humidity_pct: float
    wind_kmh: float
    rain_mm: float

    def validate(self) -> None:
        if not (-65 <= self.temperature_c <= 65):
            raise ValueError("Temperature must be between -65 and 65 °C")
        if not (0 <= self.humidity_pct <= 100):
            raise ValueError("Humidity must be 0–100%")
        if not (0 <= self.wind_kmh <= 250):
            raise ValueError("Wind speed must be 0–250 km/h")
        if not (0 <= self.rain_mm <= 500):
            raise ValueError("Rainfall must be 0–500 mm")

def score(weather: WeatherDay) -> dict[str, Any]:
    """Deterministic illustrative index, from four observable meteorological factors.

    Components normalized to [0, 1]. Lower RH and lower rain = more conducive
    to fire weather. Rain relief is an immediate-day proxy, NOT fuel moisture.
    This method intentionally omits fuels, slope, ignition, lightning and suppression.
    """
    weather.validate()
    dryness = 1 - normalized(weather.humidity_pct, 15, 85)
    wind = normalized(weather.wind_kmh, 5, 65)
    heat = normalized(weather.temperature_c, 10, 43)
    rain_relief = 1 - normalized(weather.rain_mm, 0, 12)
    components = {"dryness": dryness, "wind": wind, "heat": heat, "rain_deficit": rain_relief}
    weights = {"dryness": 0.36, "wind": 0.30, "heat": 0.19, "rain_deficit": 0.15}
    raw = sum(components[k] * weights[k] for k in components)
    # Interaction: coincidence of desiccation and wind can elevate fire weather.
    interaction = 0.12 * dryness * wind
    value = round(100 * clamp(raw + interaction), 1)
    level = "Low" if value < 25 else "Moderate" if value < 45 else "Elevated" if value < 65 else "High" if value < 80 else "Very high"
    return {
        "date": weather.day, "score": value, "level": level,
        "inputs": asdict(weather),
        "factors": {key: round(100 * component, 1) for key, component in components.items()},
        "contributions": {key: round(100 * weights[key] * components[key], 1) for key in components},
        "interaction": round(100 * interaction, 1),
    }

def demo_weather(region_id: str, n: int = 7) -> list[WeatherDay]:
    """Synthetic, repeatable data for UI demonstration ONLY; not observed weather."""
    if region_id not in REGIONS:
        raise KeyError(region_id)
    offset = list(REGIONS).index(region_id)
    first = date.today()
    days = []
    for i in range(n):
        phase = i * 0.78 + offset * 0.95
        temp = 24.0 + 6.5 * sin(phase) + (offset % 4) * 1.3
        rh = 37.0 + 18 * sin(phase + 1.8) - (offset % 3) * 3
        wind = 16 + 14 * (1 + sin(phase + 2.9)) / 2 + (offset % 4) * 3
        rain = 3.4 if i == 5 and offset % 3 == 0 else 0.0
        days.append(WeatherDay(str(first + timedelta(days=i)), round(temp, 1), round(clamp(rh, 8, 96), 1), round(wind, 1), rain))
    return days

def describe_region(region_id: str, days: list[WeatherDay], source: str) -> dict[str, Any]:
    forecasts = [score(d) for d in days]
    current = forecasts[0]
    top = max(forecasts, key=lambda item: item["score"])
    return {
        "id": region_id, **REGIONS[region_id], "source": source, "current": current,
        "forecast": forecasts, "peak_score": top["score"], "peak_date": top["date"],
        "average_score": round(sum(x["score"] for x in forecasts) / len(forecasts), 1),
    }

if __name__ == "__main__":
    for region in REGIONS:
        print(region, score(demo_weather(region)[0])["score"])
