"""Cuadrícula de celdas de 250 m sobre la ciudad de Valencia."""

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import box

from src.config import CELL_SIZE_M, CITY_BBOX, CRS, CRS_WEB


def build_grid() -> gpd.GeoDataFrame:
    """Devuelve una celda por fila: zone_id, row, col, name y geometry en EPSG:25830."""
    to_utm = Transformer.from_crs(CRS_WEB, CRS, always_xy=True)
    xmin, ymin = to_utm.transform(CITY_BBOX[0], CITY_BBOX[1])
    xmax, ymax = to_utm.transform(CITY_BBOX[2], CITY_BBOX[3])

    # Se alinea al múltiplo de 250 m para que la cuadrícula sea estable entre ejecuciones
    xmin, ymin = np.floor(xmin / CELL_SIZE_M) * CELL_SIZE_M, np.floor(ymin / CELL_SIZE_M) * CELL_SIZE_M
    xs = np.arange(xmin, xmax, CELL_SIZE_M)
    ys = np.arange(ymin, ymax, CELL_SIZE_M)

    records = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            records.append(
                {
                    "zone_id": row * len(xs) + col + 1,
                    "row": row,
                    "col": col,
                    "name": f"Celda {row:02d}-{col:02d}",
                    "geometry": box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M),
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS)


def as_matrix(zones: gpd.GeoDataFrame, column: str) -> np.ndarray:
    """Coloca una columna de la cuadrícula en una matriz (row, col) para cálculos de vecindad."""
    matrix = np.full((zones["row"].max() + 1, zones["col"].max() + 1), np.nan)
    matrix[zones["row"], zones["col"]] = zones[column]
    return matrix


def slope_from_altitude(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Pendiente media en % a partir de la altitud de las celdas vecinas."""
    altitude = as_matrix(zones, "altitude_avg")
    dy, dx = np.gradient(altitude, CELL_SIZE_M)
    slope = np.hypot(dx, dy) * 100
    return slope[zones["row"], zones["col"]]
