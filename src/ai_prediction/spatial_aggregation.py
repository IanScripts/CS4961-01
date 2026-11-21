
import math

from shapely.geometry import box

from src.ai_prediction.grid import CELL_SIZE


def get_cell_polygon(ix, iy):
    # Build our existing 10 km grid cell in EPSG:5070
    return box(
        ix * CELL_SIZE,
        iy * CELL_SIZE,
        (ix + 1) * CELL_SIZE,
        (iy + 1) * CELL_SIZE
    )


def aggregate_pixels(cell_geometry, pixels, min_coverage=0.95):
    """
    Calculate an area-weighted weather value for one grid cell.

    cell_geometry: target polygon in EPSG:5070

    pixels: list of dictionaries:
        {
            "geometry": projected pixel polygon,
            "value": weather measurement
        }

    Returns:
        value, coverage_fraction, contributing_pixels

    All geometries must use the same projected CRS.
    Pixel footprints must not overlap each other.
    """

    if cell_geometry.is_empty or cell_geometry.area <= 0:
        raise ValueError("Invalid target cell geometry")

    if not 0 < min_coverage <= 1:
        raise ValueError("Invalid minimum coverage")

    weighted_sum = 0.0
    covered_area = 0.0
    contributing_pixels = 0

    for pixel in pixels:
        value = pixel["value"]

        # Missing weather values do not contribute
        if value is None:
            continue

        value = float(value)

        if not math.isfinite(value):
            continue

        geometry = pixel["geometry"]

        if geometry.is_empty or geometry.area <= 0:
            raise ValueError("Invalid weather pixel geometry")

        # Determine how much of the pixel lies in our cell
        overlap_area = cell_geometry.intersection(geometry).area

        if overlap_area <= 0:
            continue

        weighted_sum += value * overlap_area
        covered_area += overlap_area
        contributing_pixels += 1

    coverage = covered_area / cell_geometry.area

    # Overlapping raster footprints would double-count area
    if coverage > 1.000001:
        raise ValueError("Overlapping weather pixel footprints")

    if coverage < min_coverage:
        return {
            "value": None,
            "coverage_fraction": coverage,
            "contributing_pixels": contributing_pixels
        }

    return {
        "value": weighted_sum / covered_area,
        "coverage_fraction": min(coverage, 1.0),
        "contributing_pixels": contributing_pixels
    }
