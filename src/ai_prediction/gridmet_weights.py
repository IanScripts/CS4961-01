"""Reusable area-overlap weights between gridMET latitude/longitude pixels and EPSG:5070 cells.

No weather values are included in the cache. Cache identity includes the complete
lat/lon coordinates, target cell IDs, cell size, CRS choices, and algorithm version.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import box
from shapely import transform
from shapely.strtree import STRtree

from src.ai_prediction.grid import CELL_SIZE
from src.ai_prediction.gridmet_raster import coordinate_edges

ALGORITHM_VERSION = "gridmet_pixel_overlap_v1"


@dataclass(frozen=True)
class OverlapWeights:
    cells: tuple[tuple[int, int], ...]
    raster_shape: tuple[int, int]
    row_ids: np.ndarray       # target cell index for each nonzero overlap
    pixel_ids: np.ndarray     # flattened raster pixel index for each overlap
    areas_m2: np.ndarray     # intersection area in EPSG:5070 square meters

    def aggregate(self, values, min_coverage=0.95):
        """Area-weight a (lat, lon) or (time, lat, lon) array.

        Returns (means, valid_coverage, pixel_counts), arrays shaped (cells,)
        or (time, cells). NaN/inf pixels are excluded from coverage. No partial
        observations are silently rescaled to full-cell measurements.
        """
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        values = np.asarray(values, dtype=np.float64)
        single_day = values.ndim == 2
        if single_day:
            values = values[np.newaxis, :, :]
        if values.ndim != 3 or tuple(values.shape[-2:]) != self.raster_shape:
            raise ValueError(f"Expected (..., {self.raster_shape[0]}, {self.raster_shape[1]})")

        n_days, n_cells = values.shape[0], len(self.cells)
        means = np.full((n_days, n_cells), np.nan)
        coverage = np.zeros((n_days, n_cells))
        counts = np.zeros((n_days, n_cells), dtype=np.int32)
        total_area = float(CELL_SIZE) ** 2
        for day in range(n_days):
            flat = values[day].reshape(-1)
            chosen = flat[self.pixel_ids]
            valid = np.isfinite(chosen)
            rows, areas, observed = self.row_ids[valid], self.areas_m2[valid], chosen[valid]
            weighted_sum = np.bincount(rows, weights=areas * observed, minlength=n_cells)
            valid_area = np.bincount(rows, weights=areas, minlength=n_cells)
            counts[day] = np.bincount(rows, minlength=n_cells).astype(np.int32)
            coverage[day] = valid_area / total_area
            if np.any(coverage[day] > 1.000001):
                raise ValueError("Overlapping gridMET pixel footprints produced >100% coverage")
            good = (coverage[day] >= min_coverage) & (valid_area > 0)
            means[day, good] = weighted_sum[good] / valid_area[good]
        if single_day:
            return means[0], coverage[0], counts[0]
        return means, coverage, counts


def _from_projected_polygons(cells, polygons, raster_shape):
    """Shared geometry implementation, also suitable for synthetic tests."""
    cells = tuple(tuple(map(int, cell)) for cell in cells)
    if len(set(cells)) != len(cells):
        raise ValueError("Duplicate target grid cells")
    if len(polygons) != int(np.prod(raster_shape)):
        raise ValueError("Raster geometry does not match raster shape")
    tree = STRtree(polygons)
    rows, cols, areas = [], [], []
    for cell_id, (ix, iy) in enumerate(cells):
        target = box(ix * CELL_SIZE, iy * CELL_SIZE,
                     (ix + 1) * CELL_SIZE, (iy + 1) * CELL_SIZE)
        for pixel_id in tree.query(target):
            overlap = target.intersection(polygons[int(pixel_id)]).area
            if overlap > 0:
                rows.append(cell_id)
                cols.append(int(pixel_id))
                areas.append(overlap)
    return OverlapWeights(
        cells, tuple(map(int, raster_shape)),
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(areas, dtype=np.float64),
    )


def build_weights(latitudes, longitudes, cells):
    """Construct native gridMET pixel footprints, reproject, intersect once."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    lat_edges = coordinate_edges(latitudes)
    lon_edges = coordinate_edges(longitudes)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    polygons = []
    for row in range(len(latitudes)):
        south, north = sorted((lat_edges[row], lat_edges[row + 1]))
        for col in range(len(longitudes)):
            west, east = sorted((lon_edges[col], lon_edges[col + 1]))
            polygons.append(
                transform(
                    box(west, south, east, north),
                    transformer.transform,
                    interleaved=False
                )
            )
    return _from_projected_polygons(cells, polygons, (len(latitudes), len(longitudes)))


def cache_key(latitudes, longitudes, cells):
    digest = hashlib.sha256()
    digest.update(ALGORITHM_VERSION.encode("utf-8"))
    digest.update(f"EPSG:4326|EPSG:5070|{CELL_SIZE}".encode("ascii"))
    for vector in (np.asarray(latitudes, dtype="<f8"), np.asarray(longitudes, dtype="<f8")):
        digest.update(str(vector.shape).encode("ascii"))
        digest.update(vector.tobytes())
    digest.update(json.dumps([list(map(int, c)) for c in cells], separators=(",", ":")).encode())
    return digest.hexdigest()


def load_or_build_weights(latitudes, longitudes, cells, cache_directory):
    """Return (weights, reused_cache, path); guard against stale geometry caches."""
    cells = tuple(tuple(map(int, cell)) for cell in cells)
    key = cache_key(latitudes, longitudes, cells)
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / f"gridmet_weights_{key[:20]}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["digest"].item()) != key:
                raise ValueError("Spatial weight cache identity mismatch")
            weights = OverlapWeights(
                cells, (len(latitudes), len(longitudes)),
                stored["row_ids"], stored["pixel_ids"], stored["areas_m2"],
            )
        return weights, True, path
    weights = build_weights(latitudes, longitudes, cells)
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("wb") as stream:
            np.savez_compressed(stream, digest=np.array(key),
                                row_ids=weights.row_ids, pixel_ids=weights.pixel_ids,
                                areas_m2=weights.areas_m2)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return weights, False, path
