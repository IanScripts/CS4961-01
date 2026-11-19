import unittest
from datetime import date, timedelta

import pandas as pd

from scripts.build_multicell_weather_pilot import (
    build_joined_rows,
    check_weather_frame,
)


class TestMulticellWeatherPilot(unittest.TestCase):
    def setUp(self):
        self.start = date(2018, 6, 23)
        self.end = date(2018, 7, 3)
        dates = pd.date_range(self.start, self.end, freq="D")
        self.frame = pd.DataFrame({
            "tmmx (K)": [315.0] * len(dates),
            "rmin (%)": [10.0] * len(dates),
            "vs (m/s)": [4.0] * len(dates),
            "pr (mm)": [2.0] * len(dates),
        }, index=dates)

    def test_weather_validation_and_seven_day_rainfall(self):
        weather = check_weather_frame(self.frame, self.start, self.end)
        rows = build_joined_rows([
            {"grid_cell": "-184_136", "date": "2018-07-01",
             "fire_count": 0, "fire_label": 0},
            {"grid_cell": "-184_136", "date": "2018-07-02",
             "fire_count": 1, "fire_label": 1},
        ], {"-184_136": weather})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["weather_date_gridmet"], "2018-06-29")
        self.assertEqual(rows[0]["precip_previous_7d_mm"], 14.0)
        self.assertEqual(rows[1]["fire_label"], 1)

    def test_missing_date_rejected(self):
        missing = self.frame.drop(pd.Timestamp("2018-06-26"))
        with self.assertRaisesRegex(ValueError, "Missing gridMET day"):
            check_weather_frame(missing, self.start, self.end)

    def test_wrong_units_rejected(self):
        wrong = self.frame.rename(columns={"tmmx (K)": "tmmx (F)"})
        with self.assertRaisesRegex(ValueError, "Expected gridMET column"):
            check_weather_frame(wrong, self.start, self.end)


if __name__ == "__main__":
    unittest.main()
