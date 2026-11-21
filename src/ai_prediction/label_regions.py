
import geopandas as gpd

import math
from shapely.geometry import box

from shapely.geometry import Point
from pyproj import Transformer

from src.ai_prediction.grid import CELL_SIZE, projected_cell_indices


# Nine counties selected for our Southern California study
COUNTY_FIPS = {
    "06037": "Los Angeles",
    "06111": "Ventura",
    "06083": "Santa Barbara",
    "06059": "Orange",
    "06065": "Riverside",
    "06071": "San Bernardino",
    "06073": "San Diego",
    "06025": "Imperial",
    "06029": "Kern"
}

GRID_CRS = "EPSG:5070"


def load_county_regions(boundary_path):
    # Read the Census county boundary dataset
    counties = gpd.read_file(boundary_path)

    if counties.crs is None:
        raise ValueError("County boundaries are missing a CRS")

    if "GEOID" not in counties.columns:
        raise ValueError("County boundaries are missing GEOID")

    # Select only the counties in our study area
    regions = counties[
        counties["GEOID"].isin(COUNTY_FIPS)
    ].copy()

    # Check that every required county was found
    found = set(regions["GEOID"])
    missing = set(COUNTY_FIPS) - found

    if missing:
        raise ValueError(
            f"Missing county boundaries: {sorted(missing)}"
        )

    # Convert coordinates to our 10 km grid projection
    regions = regions.to_crs(GRID_CRS)

    # Keep output ordering consistent
    regions = regions.sort_values("GEOID").reset_index(drop=True)

    return regions


def enumerate_intersecting_cells(geometry):
    # Get the projected county boundaries
    min_x, min_y, max_x, max_y = geometry.bounds

    # Find the grid columns and rows to examine
    start_x = math.floor(min_x / CELL_SIZE)
    end_x = math.ceil(max_x / CELL_SIZE)

    start_y = math.floor(min_y / CELL_SIZE)
    end_y = math.ceil(max_y / CELL_SIZE)

    cells = []

    for ix in range(start_x, end_x):
        for iy in range(start_y, end_y):

            # Create one 10 km square
            cell = box(
                ix * CELL_SIZE,
                iy * CELL_SIZE,
                (ix + 1) * CELL_SIZE,
                (iy + 1) * CELL_SIZE
            )

            # Ignore cells touching only at an edge
            if geometry.intersection(cell).area > 0:
                cells.append((ix, iy))

    return sorted(cells)


def exclude_la_buffer(candidate_cells, la_geometry, buffer_meters=10000):
    # Exclude entire grid cells within 10 km of Los Angeles
    eligible = []

    for ix, iy in sorted(candidate_cells):
        cell = box(
            ix * CELL_SIZE,
            iy * CELL_SIZE,
            (ix + 1) * CELL_SIZE,
            (iy + 1) * CELL_SIZE
        )

        # Distance is measured in projected meters
        distance = cell.distance(la_geometry)

        if distance > buffer_meters:
            eligible.append((ix, iy))

    return eligible


# FPA-FOD wildfire coordinates are geographic coordinates.
# Transform them into the grid's projected coordinate system.
fire_transformer = Transformer.from_crs(
    "EPSG:4269",
    GRID_CRS,
    always_xy=True
)


def assign_occurrence(latitude, longitude, regions):
    # Convert the wildfire's coordinates to projected meters
    x, y = fire_transformer.transform(longitude, latitude)
    point = Point(x, y)

    # Find the county containing the wildfire
    matches = regions[regions.geometry.covers(point)]

    if len(matches) != 1:
        return None

    county = matches.iloc[0]

    # Assign the wildfire to its 10 km grid cell
    ix, iy = projected_cell_indices(x, y)

    return {
        "county_fips": county["GEOID"],
        "county_name": county["NAME"],
        "grid_cell": f"{ix}_{iy}",
        "ix": ix,
        "iy": iy
    }
