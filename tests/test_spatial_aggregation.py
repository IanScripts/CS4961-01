
import unittest

from shapely.geometry import box

from src.ai_prediction.spatial_aggregation import (
    get_cell_polygon,
    aggregate_pixels
)


class TestSpatialAggregation(unittest.TestCase):

    def test_cell_geometry(self):
        cell = get_cell_polygon(-184, 136)

        self.assertEqual(cell.area, 100_000_000)

    def test_area_weighted_average(self):
        cell = box(0, 0, 10000, 10000)

        pixels = [
            {
                "geometry": box(0, 0, 2500, 10000),
                "value": 300
            },
            {
                "geometry": box(2500, 0, 10000, 10000),
                "value": 310
            }
        ]

        result = aggregate_pixels(cell, pixels)

        self.assertAlmostEqual(result["value"], 307.5)
        self.assertAlmostEqual(
            result["coverage_fraction"], 1.0
        )
        self.assertEqual(result["contributing_pixels"], 2)

    def test_insufficient_coverage(self):
        cell = box(0, 0, 10000, 10000)

        pixels = [
            {
                "geometry": box(0, 0, 5000, 10000),
                "value": 300
            }
        ]

        result = aggregate_pixels(cell, pixels)

        self.assertIsNone(result["value"])
        self.assertAlmostEqual(
            result["coverage_fraction"], 0.5
        )

    def test_missing_weather(self):
        cell = box(0, 0, 10000, 10000)

        pixels = [
            {
                "geometry": cell,
                "value": None
            }
        ]

        result = aggregate_pixels(cell, pixels)

        self.assertIsNone(result["value"])
        self.assertEqual(result["coverage_fraction"], 0)

    def test_overlapping_pixels_rejected(self):
        cell = box(0, 0, 10000, 10000)

        pixels = [
            {"geometry": cell, "value": 300},
            {"geometry": cell, "value": 310}
        ]

        with self.assertRaises(ValueError):
            aggregate_pixels(cell, pixels)


if __name__ == "__main__":
    unittest.main()
