"""Unit tests for offline IEM hourly normalization; no API requests."""
import unittest

import pandas as pd

from src.data_processing.iem_hourly import normalize_iem_hourly


class TestIemHourly(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            {"station_id": "AAT", "station_name": "Alturas", "valid": "2026-10-01 00:05:00+00:00", "tmpf": 50, "relh": 30, "sknt": 4, "drct": 100, "p01i": 0.1, "gust": 7},
            # Later report with absent temperature must not erase valid temperature.
            {"station_id": "AAT", "station_name": "Alturas", "valid": "2026-10-01 00:45:00+00:00", "tmpf": None, "relh": 20, "sknt": 6, "drct": None, "p01i": None, "gust": None},
            {"station_id": "BUR", "station_name": "Burbank", "valid": "2026-10-01 01:40:00+00:00", "tmpf": 68, "relh": 40, "sknt": 0, "drct": 0, "p01i": 0, "gust": None},
        ])

    def test_missing_hour_grid_and_provenance(self):
        out = normalize_iem_hourly(self.sample(), ["AAT", "BUR"], "2026-10-01", "2026-10-01")
        self.assertEqual(len(out), 48)
        aat = out[out.station_id == "AAT"].reset_index(drop=True)
        bur = out[out.station_id == "BUR"].reset_index(drop=True)
        self.assertEqual(int(aat.report_count.sum()), 2)
        self.assertEqual(int(aat.missing_hour.sum()), 23)
        self.assertEqual(int(bur.missing_hour.sum()), 23)
        self.assertEqual(aat.iloc[0].data_type, "observed")
        self.assertTrue(pd.isna(aat.iloc[1].temperature_f))
        self.assertTrue(bool(aat.iloc[1].missing_hour))
        self.assertEqual(int(aat.iloc[1].report_count), 0)

    def test_latest_valid_measurements_and_no_precip_sum(self):
        out = normalize_iem_hourly(self.sample(), ["AAT"], "2026-10-01", "2026-10-01")
        x = out.iloc[0]
        self.assertEqual(x.temperature_f, 50)
        self.assertEqual(x.relative_humidity_pct, 20)
        self.assertEqual(x.wind_speed_knots, 6)
        self.assertEqual(x.wind_direction_deg, 100)
        self.assertEqual(x.precipitation_report_1h_in, 0.1)
        self.assertAlmostEqual(x.wind_speed_mph, 6*1.150779448)
        self.assertEqual(x.report_count, 2)

    def test_zero_precip_is_not_missing(self):
        out = normalize_iem_hourly(self.sample(), ["BUR"], "2026-10-01", "2026-10-01")
        x = out.iloc[1]
        self.assertEqual(x.precipitation_report_1h_in, 0)
        self.assertFalse(bool(x.missing_precipitation_report_1h_in))
        self.assertFalse(bool(x.missing_hour))

    def test_unavailable_station_rejected(self):
        with self.assertRaisesRegex(ValueError, "No input observations"):
            normalize_iem_hourly(self.sample(), ["XXX"])

    def test_invalid_numeric_rejected(self):
        bad = self.sample()
        bad.loc[0, "relh"] = 150
        with self.assertRaisesRegex(ValueError, "relh"):
            normalize_iem_hourly(bad)

    def test_daylight_saving_offsets_preserved(self):
        df = pd.DataFrame([
            {"station_id": "BUR", "valid": "2026-11-01T08:40:00+00:00", "tmpf": 60},
            {"station_id": "BUR", "valid": "2026-11-01T09:40:00+00:00", "tmpf": 59},
        ])
        out = normalize_iem_hourly(df, ["BUR"], "2026-11-01", "2026-11-01")
        self.assertIn("2026-11-01T01:00:00-07:00", out.time_local.tolist())
        self.assertIn("2026-11-01T01:00:00-08:00", out.time_local.tolist())


if __name__ == "__main__":
    unittest.main()
