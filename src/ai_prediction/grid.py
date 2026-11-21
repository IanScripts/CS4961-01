
import math
from pyproj import Transformer

# Convert longitude and latitude into projected meters
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:5070",
    always_xy=True
)

CELL_SIZE = 10000  # 10 kilometers

def projected_cell_indices(x, y):
    ix = math.floor(x / CELL_SIZE)
    iy = math.floor(y / CELL_SIZE)

    return ix, iy

def get_grid_cell(latitude, longitude):
    x, y = transformer.transform(longitude, latitude)

    ix, iy = projected_cell_indices(x, y)

    return f"{ix}_{iy}"
