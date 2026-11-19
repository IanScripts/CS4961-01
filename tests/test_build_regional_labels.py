
import csv
import tempfile
import unittest

from datetime import date
from pathlib import Path

from src.ai_prediction.build_regional_labels import (
    load_config,
    build_cell_population,
    generate_pilot_dataset,
    file_sha256
)


class TestRegionalDatasetBuilder(unittest.TestCase):

    def test_real_positive_and_negative_labels(self):
        config = load_config("config/ai/regions-v1.json")

        fire_data = Path(config["sources"]["wildfires"])
        boundaries = Path(config["sources"]["counties"])

        # This is an optional real-data integration test
        if not fire_data.is_file() or not boundaries.is_file():
            self.skipTest("USDA or Census dataset not downloaded")

        regions, eligible_cells, la_cells = build_cell_population(config)

        # Known positive Riverside cell from our USDA inspection
        pilot_cells = [(-184, 136)]

        self.assertIn(pilot_cells[0], eligible_cells)

        # Use the dates with verified wildfire reports
        config["pilot"]["windows"] = [
            {
                "name": "positive_control",
                "start_date": "2018-07-01",
                "end_date": "2018-07-04"
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            generate_pilot_dataset(
                config,
                regions,
                eligible_cells,
                pilot_cells,
                output_dir=temp_dir
            )

            label_path = (
                Path(temp_dir) / config["outputs"]["fire_labels"]
            )

            with label_path.open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))

            # One cell across four days
            self.assertEqual(len(rows), 4)

            daily = {
                row["date"]: row
                for row in rows
            }

            self.assertEqual(
                int(daily["2018-07-02"]["fire_count"]), 1
            )
            self.assertEqual(
                int(daily["2018-07-03"]["fire_count"]), 2
            )

            self.assertEqual(
                int(daily["2018-07-02"]["fire_label"]), 1
            )
            self.assertEqual(
                int(daily["2018-07-03"]["fire_label"]), 1
            )

            self.assertEqual(
                int(daily["2018-07-01"]["fire_count"]), 0
            )

            # Verify the manifest records the CSV checksum
            import json

            manifest_path = (
                Path(temp_dir) / config["outputs"]["manifest"]
            )

            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)

            self.assertEqual(
                manifest["outputs"]["fire_labels"]["sha256"],
                file_sha256(label_path)
            )


if __name__ == "__main__":
    unittest.main()
