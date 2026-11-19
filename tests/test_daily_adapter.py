"""Offline tests for the cross-project bridge; inputs are synthetic test fixtures."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_open_meteo_csv import create_daily_scores, expected_hours


class BridgeTests(unittest.TestCase):
    def sample(self, start_utc, n=24):
        ts = pd.date_range(start_utc, periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            "time": ts.strftime("%Y-%m-%dT%H:%M"),
            "location": ["Burbank"]*n,
            "temperature_2m": [77.0]*n,
            "relative_humidity_2m": [25.0]*n,
            "wind_speed_10m": [12.0]*n,
            "precipitation": [0.0]*n,
            "data_type": ["historical_reanalysis"]*n,
            "source": ["Open-Meteo Historical API"]*n,
        })

    def test_complete_local_day(self):
        # Midnight California standard time is 08:00 UTC.
        result = create_daily_scores(self.sample("2026-11-04T08:00:00Z"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["local_date"], "2026-11-04")
        self.assertGreater(result.iloc[0]["experimental_score"], 0)
        self.assertEqual(result.iloc[0]["metric_kind"], "UNCALIBRATED_EXPERIMENTAL_WEATHER_SCORE")

    def test_incomplete_day_is_not_scored(self):
        result = create_daily_scores(self.sample("2026-11-04T08:00:00Z", 22))
        self.assertTrue(result.empty)

    def test_missing_or_duplicates_not_scored(self):
        src = self.sample("2026-11-04T08:00:00Z")
        src.loc[5, "wind_speed_10m"] = None
        self.assertTrue(create_daily_scores(src).empty)
        src = self.sample("2026-11-04T08:00:00Z")
        src.loc[5,"time"] = src.loc[4,"time"]
        self.assertTrue(create_daily_scores(src).empty)

    def test_unmixed_provenance(self):
        src = self.sample("2026-11-04T08:00:00Z")
        src.loc[5,"data_type"]="forecast"
        self.assertTrue(create_daily_scores(src).empty)

    def test_dst_day_lengths(self):
        self.assertEqual(expected_hours(pd.Timestamp("2026-03-08")), 23)
        self.assertEqual(expected_hours(pd.Timestamp("2026-11-01")), 25)

    def test_bad_columns(self):
        with self.assertRaisesRegex(ValueError,"Missing required"):
            create_daily_scores(pd.DataFrame({"time":[]}))

if __name__ == '__main__': unittest.main()
