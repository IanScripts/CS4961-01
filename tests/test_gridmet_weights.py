import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box

from src.ai_prediction.grid import CELL_SIZE
from src.ai_prediction.gridmet_weights import (
    _from_projected_polygons, cache_key, load_or_build_weights,
)
from src.ai_prediction.spatial_aggregation import aggregate_pixels


class TestGridmetWeights(unittest.TestCase):
    def setUp(self):
        self.cell = (0, 0)
        self.polygons = [box(0, 0, 2500, 10000), box(2500, 0, 10000, 10000)]
        self.weights = _from_projected_polygons([self.cell], self.polygons, (1, 2))

    def test_matches_existing_area_weighted_function(self):
        old = aggregate_pixels(box(0, 0, CELL_SIZE, CELL_SIZE), [
            {"geometry": polygon, "value": value}
            for polygon, value in zip(self.polygons, [300, 310])
        ])
        means, cover, counts = self.weights.aggregate([[300, 310]])
        self.assertAlmostEqual(means[0], old["value"])
        self.assertAlmostEqual(cover[0], old["coverage_fraction"])
        self.assertEqual(counts[0], 2)

    def test_multiple_dates(self):
        means, cover, counts = self.weights.aggregate([[[300, 310]], [[320, 300]]])
        np.testing.assert_allclose(means[:, 0], [307.5, 305.0])
        np.testing.assert_allclose(cover[:, 0], [1, 1])
        np.testing.assert_array_equal(counts[:, 0], [2, 2])

    def test_missing_data_affects_coverage(self):
        means, cover, counts = self.weights.aggregate([[np.nan, 310]])
        self.assertTrue(np.isnan(means[0]))
        self.assertAlmostEqual(cover[0], 0.75)
        self.assertEqual(counts[0], 1)
        means, _, _ = self.weights.aggregate([[np.nan, 310]], min_coverage=0.70)
        self.assertAlmostEqual(means[0], 310.0)

    def test_duplicate_cells_rejected(self):
        with self.assertRaises(ValueError):
            _from_projected_polygons([(0, 0), (0, 0)], self.polygons, (1, 2))

    def test_rejects_wrong_raster_shape(self):
        with self.assertRaises(ValueError):
            self.weights.aggregate(np.ones((2, 2)))

    def test_cache_reuse_and_coordinates_key(self):
        latitudes = [33.9, 33.86]
        longitudes = [-116.5, -116.46]
        cells = [(-184, 136)]
        with tempfile.TemporaryDirectory() as tmp:
            first, reused, path = load_or_build_weights(latitudes, longitudes, cells, tmp)
            self.assertFalse(reused)
            self.assertTrue(Path(path).exists())
            second, reused, _ = load_or_build_weights(latitudes, longitudes, cells, tmp)
            self.assertTrue(reused)
            np.testing.assert_array_equal(first.areas_m2, second.areas_m2)
            self.assertNotEqual(cache_key(latitudes, longitudes, cells),
                                cache_key(latitudes, [-116.5, -116.45], cells))


if __name__ == "__main__":
    unittest.main()
