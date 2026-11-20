
import math

import numpy as np
import xarray as xr

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from src.ai_prediction.spatial_aggregation import (
    aggregate_pixels,
    get_cell_polygon
)


def coordinate_edges(coordinates):
    """
    Estimate pixel boundaries from evenly spaced
    latitude or longitude center coordinates.
    """
    centers = np.asarray(coordinates, dtype=float)

    if len(centers) < 2:
        raise ValueError("At least two coordinates required")

    differences = np.diff(centers)

    if not (
        np.all(differences > 0)
        or np.all(differences < 0)
    ):
        raise ValueError("Coordinates must be monotonic")

    edges = np.empty(len(centers) + 1)

    edges[1:-1] = (centers[:-1] + centers[1:]) / 2

    edges[0] = centers[0] - differences[0] / 2
    edges[-1] = centers[-1] + differences[-1] / 2

    return edges


def aggregate_raster_day(
    dataset,
    variable_name,
    weather_date,
    ix,
    iy
):
    """
    Calculate area-weighted weather for one
    EPSG:5070 wildfire cell and one gridMET day.
    """
    data = dataset[variable_name].sel(
        day=weather_date
    )

    if data.dims != ("lat", "lon"):
        raise ValueError(
            f"Unexpected raster dimensions: {data.dims}"
        )

    latitude_edges = coordinate_edges(
        data["lat"].values
    )

    longitude_edges = coordinate_edges(
        data["lon"].values
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:5070",
        always_xy=True
    )

    target_cell = get_cell_polygon(ix, iy)

    pixels = []

    for row in range(data.sizes["lat"]):
        south = min(
            latitude_edges[row],
            latitude_edges[row + 1]
        )

        north = max(
            latitude_edges[row],
            latitude_edges[row + 1]
        )

        for col in range(data.sizes["lon"]):
            west = min(
                longitude_edges[col],
                longitude_edges[col + 1]
            )

            east = max(
                longitude_edges[col],
                longitude_edges[col + 1]
            )

            # Construct the native geographic pixel
            geographic_pixel = box(
                west, south, east, north
            )

            # Transform polygon to our wildfire grid CRS
            projected_pixel = transform(
                transformer.transform,
                geographic_pixel
            )

            # Skip pixels that do not intersect the cell
            if not projected_pixel.intersects(target_cell):
                continue

            value = float(
                data.values[row, col]
            )

            pixels.append({
                "geometry": projected_pixel,
                "value": value if math.isfinite(value) else None
            })

    return aggregate_pixels(
        target_cell,
        pixels,
        min_coverage=0.95
    )


def get_center_sample(
    dataset,
    variable_name,
    weather_date,
    ix,
    iy
):
    """
    Retrieve the gridMET pixel nearest
    to the center of our wildfire cell.
    """
    from src.ai_prediction.grid import CELL_SIZE

    x = (ix + 0.5) * CELL_SIZE
    y = (iy + 0.5) * CELL_SIZE

    transformer = Transformer.from_crs(
        "EPSG:5070",
        "EPSG:4326",
        always_xy=True
    )

    longitude, latitude = transformer.transform(x, y)

    sample = dataset[variable_name].sel(
        day=weather_date,
        lat=latitude,
        lon=longitude,
        method="nearest"
    )

    return float(sample.item())
