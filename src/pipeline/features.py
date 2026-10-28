"""Cálculo de variables: lluvia por celda, acumulados y tabla de entrenamiento."""

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import MAX_STATION_DIST_M


def nearest_station(zones: gpd.GeoDataFrame, stations: gpd.GeoDataFrame) -> pd.DataFrame:
    """Estación más cercana a cada celda y su distancia: zone_id, station_id, station_dist_m."""
    centroids = zones[["zone_id"]].set_geometry(zones.geometry.centroid)
    joined = gpd.sjoin_nearest(
        centroids, stations[["station_id", "geometry"]], distance_col="station_dist_m"
    ).drop_duplicates("zone_id")
    joined["station_dist_m"] = joined["station_dist_m"].round(0)
    joined.loc[joined["station_dist_m"] > MAX_STATION_DIST_M, "station_id"] = None
    return joined[["zone_id", "station_id", "station_dist_m"]].reset_index(drop=True)


def rain_accumulations(station_rain: pd.DataFrame) -> pd.DataFrame:
    """Añade precipitation_1h y precipitation_24h (suma móvil) por estación y episodio."""
    df = station_rain.sort_values(["station_id", "episode_id", "observed_at"]).copy()
    df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0).fillna(0)
    grouped = df.groupby(["station_id", "episode_id"])["precipitation_mm"]
    df["precipitation_1h"] = df["precipitation_mm"]
    df["precipitation_24h"] = grouped.transform(lambda s: s.rolling(24, min_periods=1).sum()).round(2)
    return df


def weather_by_zone(zone_station: pd.DataFrame, station_rain: pd.DataFrame) -> pd.DataFrame:
    """WeatherObservation: lluvia horaria de cada celda tomada de su estación asignada."""
    df = zone_station.dropna(subset=["station_id"]).merge(station_rain, on="station_id")
    df = df.sort_values(["episode_id", "observed_at", "zone_id"]).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df[
        ["id", "zone_id", "station_id", "episode_id", "observed_at", "precipitation_mm",
         "precipitation_1h", "precipitation_24h", "temperature_c"]
    ]


def episode_peaks(weather: pd.DataFrame) -> pd.DataFrame:
    """Máximos de lluvia de cada celda en cada episodio, que son las variables de entrenamiento."""
    return (
        weather.groupby(["zone_id", "episode_id"])
        .agg(precipitation_1h=("precipitation_1h", "max"), precipitation_24h=("precipitation_24h", "max"))
        .reset_index()
    )


def patricova_score(level: pd.Series) -> pd.Series:
    """Invierte el nivel PATRICOVA (1 = máxima peligrosidad) a una escala 0-6 donde más es peor."""
    return np.where(level > 0, 7 - level, 0)
