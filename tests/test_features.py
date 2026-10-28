import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.config import CELL_SIZE_M, CRS
from src.pipeline import features
from src.pipeline.grid import build_grid, slope_from_altitude


def test_grid_cells_are_250m_with_unique_ids():
    zones = build_grid()
    assert zones.crs.to_string() == CRS
    assert zones["zone_id"].is_unique
    assert np.allclose(zones.geometry.area, CELL_SIZE_M**2)


def test_rain_24h_is_rolling_sum_and_negatives_are_clipped():
    rain = pd.DataFrame({
        "station_id": "S1",
        "episode_id": 1,
        "observed_at": pd.date_range("2024-10-29", periods=30, freq="h"),
        "precipitation_mm": [-1.0] + [1.0] * 29,
    })
    result = features.rain_accumulations(rain)
    assert result["precipitation_mm"].min() == 0
    assert result["precipitation_24h"].iloc[23] == 23
    assert result["precipitation_24h"].iloc[29] == 24
    assert (result["precipitation_1h"] == result["precipitation_mm"]).all()


def test_rain_accumulation_does_not_mix_episodes():
    rain = pd.DataFrame({
        "station_id": "S1",
        "episode_id": [1, 1, 2, 2],
        "observed_at": pd.date_range("2024-01-01", periods=4, freq="h"),
        "precipitation_mm": [10.0, 10.0, 1.0, 1.0],
    })
    result = features.rain_accumulations(rain)
    assert result.loc[result["episode_id"] == 2, "precipitation_24h"].tolist() == [1.0, 2.0]


def test_patricova_score_inverts_levels():
    levels = pd.Series([0, 1, 6])
    assert list(features.patricova_score(levels)) == [0, 6, 1]


def test_nearest_station_distance():
    zones = build_grid().head(1)
    centroid = zones.geometry.centroid.iloc[0]
    stations = gpd.GeoDataFrame(
        {"station_id": ["NEAR", "FAR"]},
        geometry=[Point(centroid.x + 100, centroid.y), Point(centroid.x + 5000, centroid.y)],
        crs=CRS,
    )
    result = features.nearest_station(zones, stations)
    assert result.loc[0, "station_id"] == "NEAR"
    assert result.loc[0, "station_dist_m"] == 100


def test_slope_of_tilted_plane():
    zones = build_grid()
    zones["altitude_avg"] = zones["col"] * 2.5  # 2.5 m cada 250 m = 1 %
    assert np.allclose(slope_from_altitude(zones), 1.0)
