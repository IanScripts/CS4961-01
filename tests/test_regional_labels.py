
import unittest
import geopandas as gpd

from datetime import date
from shapely.geometry import box
from pyproj import Transformer

from src.ai_prediction.grid import projected_cell_indices
from src.ai_prediction.regional_labels import (
    aggregate_event_counts,
    iter_label_rows
)


class TestRegionalLabels(unittest.TestCase):

    def setUp(self):
        # Two synthetic counties near Southern California
        self.regions = gpd.GeoDataFrame(
            {
                "GEOID": ["06037", "06065"],
                "NAME": ["Los Angeles", "Riverside"]
            },
            geometry=[
                box(-119, 33, -118, 35),
                box(-118, 33, -117, 35)
            ],
            crs="EPSG:4269"
        ).to_crs("EPSG:5070")

        transformer = Transformer.from_crs(
            "EPSG:4269", "EPSG:5070", always_xy=True
        )

        x, y = transformer.transform(-117.5, 34.0)
        self.riverside_cell = projected_cell_indices(x, y)

        x, y = transformer.transform(-118.5, 34.0)
        self.la_cell = projected_cell_indices(x, y)

        self.eligible_cells = {self.riverside_cell}

        self.events = [
            {
                "fod_id": 1,
                "discovery_date": date(2018, 7, 1),
                "latitude": 34.0,
                "longitude": -117.5
            },
            {
                "fod_id": 2,
                "discovery_date": date(2018, 7, 1),
                "latitude": 34.0,
                "longitude": -117.5
            },
            {
                "fod_id": 3,
                "discovery_date": date(2018, 7, 2),
                "latitude": 34.0,
                "longitude": -118.5
            }
        ]

    def test_multiple_fires_one_cell(self):
        counts = aggregate_event_counts(
            self.events, self.regions, self.eligible_cells
        )

        key = (
            "non_la",
            f"{self.riverside_cell[0]}_{self.riverside_cell[1]}",
            date(2018, 7, 1)
        )

        self.assertEqual(counts[key], 2)

    def test_la_separate_from_non_la(self):
        counts = aggregate_event_counts(
            self.events, self.regions, self.eligible_cells
        )

        splits = {key[0] for key in counts}

        self.assertEqual(splits, {"non_la", "la_evaluation"})

    def test_explicit_zero_days(self):
        counts = aggregate_event_counts(
            self.events, self.regions, self.eligible_cells
        )

        rows = list(iter_label_rows(
            {
                "non_la": [self.riverside_cell],
                "la_evaluation": [self.la_cell]
            },
            date(2018, 7, 1),
            date(2018, 7, 3),
            counts
        ))

        # Two cells across three days
        self.assertEqual(len(rows), 6)

        # One cell has two fires on July 1
        self.assertEqual(rows[0]["fire_count"], 2)
        self.assertEqual(rows[0]["fire_label"], 1)

        # Confirm that zero-fire days are included
        self.assertTrue(
            any(row["fire_label"] == 0 for row in rows)
        )

    def test_duplicate_fire_ids(self):
        repeated = [self.events[0], self.events[0]]

        with self.assertRaises(ValueError):
            aggregate_event_counts(
                repeated, self.regions, self.eligible_cells
            )


if __name__ == "__main__":
    unittest.main()
