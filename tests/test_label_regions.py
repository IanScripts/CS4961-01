
import unittest
from pathlib import Path

from shapely.geometry import box

from src.ai_prediction.label_regions import (
    load_county_regions,
    enumerate_intersecting_cells,
    exclude_la_buffer,
    assign_occurrence
)


class TestLabelRegions(unittest.TestCase):

    def test_county_boundaries(self):
        boundary_path = Path(
            "wildfire-datasets/boundaries/counties_2024.zip"
        )

        if not boundary_path.exists():
            self.skipTest("Census boundary dataset not downloaded")

        regions = load_county_regions(boundary_path)

        self.assertEqual(len(regions), 9)
        self.assertEqual(regions.crs.to_epsg(), 5070)
        self.assertIn("06037", regions["GEOID"].values)
        self.assertIn("06073", regions["GEOID"].values)

    
    def test_enumerate_intersecting_cells(self):
        # Rectangle spanning three 10 km cells
        geometry = box(-5000, 0, 15000, 10000)

        cells = enumerate_intersecting_cells(geometry)

        self.assertEqual(
            cells,
            [(-1, 0), (0, 0), (1, 0)]
        )

    
    def test_exclude_la_buffer(self):
        # Synthetic LA polygon occupying the first grid cell
        la_geometry = box(0, 0, 10000, 10000)

        candidate_cells = {
            (0, 0),   # Overlaps LA
            (1, 0),   # Touches LA
            (2, 0),   # Exactly 10 km away
            (3, 0),   # 20 km away
            (-2, 0),  # Exactly 10 km away
        }

        eligible = exclude_la_buffer(
            candidate_cells,
            la_geometry
        )

        self.assertEqual(eligible, [(3, 0)])


    def test_assign_occurrence(self):
        import geopandas as gpd

        # Two synthetic geographic regions divided at longitude -118
        left_region = box(-119, 33, -118, 35)
        right_region = box(-118, 33, -117, 35)

        regions = gpd.GeoDataFrame(
            {
                "GEOID": ["06037", "06111"],
                "NAME": ["Left Region", "Right Region"]
            },
            geometry=[left_region, right_region],
            crs="EPSG:4269"
        ).to_crs("EPSG:5070")

        left_fire = assign_occurrence(34, -118.5, regions)
        right_fire = assign_occurrence(34, -117.5, regions)

        self.assertEqual(left_fire["county_fips"], "06037")
        self.assertEqual(right_fire["county_fips"], "06111")



if __name__ == "__main__":
    unittest.main()
