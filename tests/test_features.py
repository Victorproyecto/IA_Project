import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import BLOCK_SIZE_CELLS, CELL_SIZE_M, CRS, N_FOLDS
from src.pipeline import blocks, features, sources
from src.pipeline.grid import build_grid, slope_from_altitude


def test_grid_cells_have_configured_size_and_unique_ids():
    zones = build_grid()
    assert zones.crs.to_string() == CRS
    assert zones["zone_id"].is_unique
    assert np.allclose(zones.geometry.area, CELL_SIZE_M**2)


def test_patricova_score_inverts_levels():
    levels = pd.Series([0, 1, 6])
    assert list(features.patricova_score(levels)) == [0, 6, 1]


def test_slope_of_tilted_plane():
    zones = build_grid()
    zones["altitude_avg"] = zones["col"] * CELL_SIZE_M / 100  # 1 m cada 100 m = 1 %
    assert np.allclose(slope_from_altitude(zones), 1.0)


def test_channel_distance_is_zero_on_channel_and_grows_away():
    zones = build_grid().head(3)
    channel = gpd.GeoDataFrame({"nombre": ["Cauce"]}, geometry=[zones.geometry.iloc[0].centroid.buffer(1)], crs=CRS)
    distance, name = sources.channel_distance(zones, channel)
    assert distance[0] < 0.01 and distance[1] > distance[0]
    assert list(name) == ["Cauce"] * 3


def test_covered_fraction_by_group():
    """Una celda cubierta a medias por un grupo y a un cuarto por otro; la otra celda, sin cubrir."""
    from shapely.geometry import box

    zones = build_grid().head(2)
    x0, y0, x1, y1 = zones.geometry.iloc[0].bounds
    half = box(x0, y0, x0 + (x1 - x0) / 2, y1)
    quarter = box(x0 + (x1 - x0) / 2, y0, x1, y0 + (y1 - y0) / 2)
    polygons = gpd.GeoDataFrame({"grupo": ["a", "b"]}, geometry=[half, quarter], crs=CRS)
    fractions = sources._covered_fraction(zones, polygons, by="grupo")
    assert np.allclose(fractions.loc[zones.index[0], ["a", "b"]], [0.5, 0.25])
    assert np.allclose(fractions.loc[zones.index[1], ["a", "b"]], [0, 0])
    assert np.allclose(sources._covered_fraction(zones, polygons)["_fraction"], [0.75, 0])


def test_touches_marks_only_cells_in_contact():
    zones = build_grid().head(3)
    polygon = gpd.GeoDataFrame(geometry=[zones.geometry.iloc[0].centroid.buffer(10)], crs=CRS)
    assert list(sources._touches(zones, polygon)) == [1, 0, 0]


def test_depth_severity_uses_copernicus_cuts():
    depth = pd.Series([0.1, 0.5, 0.99, 1.0, 1.01, 3.0])
    assert list(sources.depth_severity(depth)) == ["Leve", "Moderada", "Moderada", "Moderada", "Grave", "Grave"]


def test_blocks_stay_in_one_fold_and_every_fold_has_both_classes():
    zones = build_grid()
    flooded = (zones["row"] < 2 * BLOCK_SIZE_CELLS).astype(int)
    block = blocks.block_ids(zones["row"], zones["col"])
    fold = blocks.assign_folds(block, flooded)
    assert fold.groupby(block).nunique().eq(1).all()
    assert set(fold) == set(range(N_FOLDS))
    assert flooded.groupby(fold).nunique().eq(2).all()
