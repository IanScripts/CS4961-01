"""Offline tests: lightweight statewide daily summaries, without downloads."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_iem_statewide_daily import aggregate_chunks, expected_local_hours


def records_for_day(date, station="AAT", missing=None):
    start = pd.Timestamp(date, tz="America/Los_Angeles").tz_convert("UTC")
    stop = pd.Timestamp((pd.Timestamp(date) + pd.Timedelta(days=1)).date(), tz="America/Los_Angeles").tz_convert("UTC")
    ts = pd.date_range(start, stop, freq="h", inclusive="left")
    rows = []
    for i, t in enumerate(ts):
        omitted = i == missing
        rows.append(dict(
            station_id=station, time_utc=t.isoformat(), report_count=0 if omitted else 1,
            source="IEM ASOS archive", data_type="observed",
            temperature_f=None if omitted else 50+i,
            relative_humidity_pct=None if omitted else 80-i,
            wind_speed_mph=None if omitted else 4+i,
            precipitation_report_1h_in=None,
        ))
    return pd.DataFrame(rows)


class DailyTests(unittest.TestCase):
    def test_dst_spring_and_fall(self):
        self.assertEqual(expected_local_hours("2025-03-09"), 23)
        self.assertEqual(expected_local_hours("2025-11-02"), 25)
        self.assertEqual(expected_local_hours("2025-03-10"), 24)

    def test_across_chunks_with_missing_reports(self):
        d = records_for_day("2025-04-01", missing=7)
        out, partial = aggregate_chunks([d.iloc[:10], d.iloc[10:]], 2025)
        self.assertEqual(partial, 0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["hour_slots"], 24)
        self.assertEqual(out.iloc[0]["hours_with_reports"], 23)
        self.assertEqual(out.iloc[0]["temperature_hours"], 23)
        self.assertAlmostEqual(out.iloc[0]["report_coverage_pct"], 95.83)
        self.assertEqual(out.iloc[0]["humidity_min_pct"], 57)
        self.assertEqual(out.iloc[0]["wind_max_mph"], 27)

    def test_partial_day_is_not_exported(self):
        d = records_for_day("2025-04-01").iloc[:20]
        out, partial = aggregate_chunks([d], 2025)
        self.assertEqual(len(out), 0)
        self.assertEqual(partial, 1)

    def test_provenance_rejected(self):
        d = records_for_day("2025-04-01")
        d["data_type"] = "forecast"
        with self.assertRaisesRegex(ValueError, "data_type"):
            aggregate_chunks([d], 2025)

    def test_all_missing_values_not_zero(self):
        d = records_for_day("2025-04-01")
        d["temperature_f"] = None
        out, _ = aggregate_chunks([d], 2025)
        self.assertTrue(pd.isna(out.loc[0, "temperature_mean_f"]))
        self.assertTrue(pd.isna(out.loc[0, "temperature_max_f"]))
        self.assertEqual(out.loc[0, "temperature_hours"], 0)

    def test_multi_station(self):
        x = pd.concat([records_for_day("2025-04-01", "AAT"), records_for_day("2025-04-01", "BUR")])
        out, partial = aggregate_chunks([x.iloc[:35], x.iloc[35:]], 2025)
        self.assertEqual(len(out), 2)
        self.assertEqual(partial, 0)
        self.assertEqual(set(out.station_id), {"AAT", "BUR"})

if __name__ == "__main__":
    unittest.main()
