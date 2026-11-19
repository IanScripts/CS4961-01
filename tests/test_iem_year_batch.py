"""Offline unit tests for streaming one-year statewide IEM processing."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_iem_year_hourly import build, completed_stations, quality_row


class TestIEMYearBatch(unittest.TestCase):
    def test_manifest_excludes_incomplete(self):
        catalog = pd.DataFrame({"station_id": ["AAT", "BUR", "CQT"]})
        manifest = pd.DataFrame({
            "station_id": ["AAT", "BUR", "CQT"],
            "year": [2025, 2025, 2025],
            "requested_start": ["2025-01-01", "2025-01-01", "2025-02-01"],
            "requested_end": ["2025-12-31", "2025-12-31", "2025-12-31"],
        })
        complete, excluded = completed_stations(manifest, catalog, 2025)
        self.assertEqual(complete, ["AAT", "BUR"])
        self.assertEqual(excluded, ["CQT"])

    def _fixture(self, root):
        input_file, manifest_file = root/"raw.csv", root/"manifest.csv"
        catalog_file, output_file = root/"catalog.csv", root/"out.csv"
        quality_file = root/"quality.csv"
        rows = [
            # Repeated report in one hour, no summing precipitation reports.
            {"station_id":"AAT","valid":"2025-01-01 00:20","tmpf":50,"relh":45,"sknt":10,"drct":90,"p01i":0.0},
            {"station_id":"AAT","valid":"2025-01-01 00:50","tmpf":53,"relh":42,"sknt":11,"drct":95,"p01i":0.1},
            {"station_id":"BUR","valid":"2025-01-02 01:00","tmpf":67,"relh":50,"sknt":6,"drct":100,"p01i":0.0},
            # 2026 is intentionally mixed in the original raw file and must be excluded.
            {"station_id":"AAT","valid":"2026-11-05 00:00","tmpf":80,"relh":25,"sknt":25,"drct":180,"p01i":0.0},
        ]
        pd.DataFrame(rows).to_csv(input_file, index=False)
        pd.DataFrame({"station_id":["AAT","BUR","CQT"]}).to_csv(catalog_file,index=False)
        pd.DataFrame({
            "station_id":["AAT","BUR"], "year":[2025,2025],
            "requested_start":["2025-01-01"]*2,
            "requested_end":["2025-12-31"]*2,
        }).to_csv(manifest_file,index=False)
        return input_file,manifest_file,catalog_file,output_file,quality_file

    def test_streaming_full_year_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._fixture(Path(d))
            n,rows=build(*paths,2025,chunk_size=100)
            self.assertEqual((n,rows),(2,17520))
            hourly = pd.read_csv(paths[3])
            summary = pd.read_csv(paths[4])
            self.assertEqual(len(hourly),17520)
            self.assertEqual(len(summary),2)
            self.assertTrue(set(hourly["station_id"])=={"AAT","BUR"})
            self.assertTrue((hourly["source"]=="IEM ASOS archive").all())
            aat=hourly[(hourly["station_id"]=="AAT")&(hourly["time_utc"].str.startswith("2025-01-01T00:"))]
            self.assertEqual(aat.iloc[0]["report_count"],2)
            self.assertAlmostEqual(float(aat.iloc[0]["precipitation_report_1h_in"]),0.1)
            self.assertEqual(int(summary.loc[summary.station_id=="AAT","hours_with_reports"].iloc[0]),1)
            self.assertEqual(int(summary.loc[summary.station_id=="BUR","hours_with_reports"].iloc[0]),1)
            self.assertEqual(hourly["missing_hour"].sum(),17518)

    def test_missing_station_does_not_overwrite_prior_output(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._fixture(Path(d))
            paths[3].write_text("previous good hourly file",encoding="utf-8")
            # Add a misleading manifest entry for a station with no raw 2025 reports.
            manifest=pd.read_csv(paths[1]); manifest.loc[len(manifest)]={
                "station_id":"CQT","year":2025,
                "requested_start":"2025-01-01","requested_end":"2025-12-31"}
            manifest.to_csv(paths[1],index=False)
            with self.assertRaisesRegex(ValueError,"absent"):
                build(*paths,2025,chunk_size=100)
            self.assertEqual(paths[3].read_text(encoding="utf-8"),"previous good hourly file")

    def test_selected_station_requires_custom_output(self):
        # Check the API can process only an explicitly requested manifest-complete station.
        with tempfile.TemporaryDirectory() as d:
            paths=self._fixture(Path(d))
            n,rows=build(*paths,2025,stations=["BUR"],chunk_size=100)
            self.assertEqual((n,rows),(1,8760))
            self.assertEqual(pd.read_csv(paths[3]).station_id.unique().tolist(),["BUR"])


if __name__ == "__main__":
    unittest.main()
