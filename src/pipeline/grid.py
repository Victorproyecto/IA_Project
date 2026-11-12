"""Cuadrícula de celdas de CELL_SIZE_M (100 m) sobre el área piloto o la prueba externa."""

import geopandas as gpd
import numpy as np
import shapely

from src.config import CELL_SIZE_M, CRS, PILOT_EXTENT

PERCENT = 100


def build_grid(extent: tuple[float, float, float, float] = PILOT_EXTENT) -> gpd.GeoDataFrame:
    """Devuelve una celda por fila: zone_id, row, col, name y geometry en EPSG:25830.

    Las celdas se numeran por filas de sur a norte y, dentro de cada fila, de oeste a este (zone_id empieza en 1).
    """
    # El área ya está en UTM y alineada a múltiplos de 100 y 250 m: la cuadrícula es estable entre ejecuciones
    xmin, ymin, xmax, ymax = extent
    xs = np.arange(xmin, xmax, CELL_SIZE_M)
    ys = np.arange(ymin, ymax, CELL_SIZE_M)
    row, col = np.divmod(np.arange(len(xs) * len(ys)), len(xs))
    x0, y0 = xs[col], ys[row]
    return gpd.GeoDataFrame(
        {
            "zone_id": row * len(xs) + col + 1,
            "row": row,
            "col": col,
            "name": [f"Celda {r:02d}-{c:02d}" for r, c in zip(row, col, strict=True)],
        },
        geometry=shapely.box(x0, y0, x0 + CELL_SIZE_M, y0 + CELL_SIZE_M),
        crs=CRS,
    )


def as_matrix(zones: gpd.GeoDataFrame, column: str) -> np.ndarray:
    """Coloca una columna de la cuadrícula en una matriz (row, col) para cálculos de vecindad."""
    matrix = np.full((zones["row"].max() + 1, zones["col"].max() + 1), np.nan)
    matrix[zones["row"], zones["col"]] = zones[column]
    return matrix


def slope_from_altitude(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Pendiente media en % a partir de la altitud de las celdas vecinas."""
    altitude = as_matrix(zones, "altitude_avg")
    dy, dx = np.gradient(altitude, CELL_SIZE_M)
    slope = np.hypot(dx, dy) * PERCENT
    return slope[zones["row"], zones["col"]]
