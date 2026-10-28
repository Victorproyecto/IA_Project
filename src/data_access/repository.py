"""Capa de acceso a datos entre el almacenamiento Parquet y el dashboard.

El dashboard solo usa estas funciones; si se migra a PostgreSQL/PostGIS, solo cambia este módulo.
"""

import json
from functools import lru_cache

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.config import (
    CRS, CRS_WEB, DATA_SOURCES_FILE, EPISODES_FILE, FLOOD_EVENTS_FILE, MODEL_CARD_FILE, PREDICTIONS_DIR,
    RISK_LEVELS, ZONES_FILE,
)


@lru_cache
def get_zones() -> gpd.GeoDataFrame:
    """Zonas en EPSG:25830 con sus atributos físicos, distrito y barrio."""
    return gpd.read_parquet(ZONES_FILE)


def get_zones_web() -> gpd.GeoDataFrame:
    """Zonas en WGS84 para el mapa."""
    return get_zones().to_crs(CRS_WEB)


@lru_cache
def get_model_card() -> dict:
    return json.loads(MODEL_CARD_FILE.read_text())


def get_data_sources() -> pd.DataFrame:
    return pd.read_csv(DATA_SOURCES_FILE)


def list_episodes() -> pd.DataFrame:
    """Episodios históricos, escenarios predefinidos y previsiones, del más reciente al más antiguo."""
    episodes = pd.read_parquet(EPISODES_FILE)
    order = {"forecast": 0, "historical": 1, "scenario": 2}
    return episodes.sort_values(["kind", "started_at"], key=lambda s: s.map(order) if s.name == "kind" else s, ascending=[True, False])


def _predictions(episode_id: int, model_version: str | None = None) -> pd.DataFrame:
    version = model_version or get_model_card()["model_version"]
    return pd.read_parquet(
        PREDICTIONS_DIR / f"model_version={version}", filters=[("episode_id", "==", int(episode_id))]
    ).assign(model_version=version)


def get_episode_hours(episode_id: int) -> list[pd.Timestamp]:
    return sorted(_predictions(episode_id)["target_time"].unique())


def get_predictions(episode_id: int, target_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """Predicción por celda a una hora concreta o, sin hora, la de mayor probabilidad del episodio."""
    df = _predictions(episode_id)
    if target_time is not None:
        df = df[df["target_time"] == pd.Timestamp(target_time)]
    else:
        df = df.sort_values("risk_probability", ascending=False).drop_duplicates("zone_id")
    df = df.copy()
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=RISK_LEVELS, ordered=True)
    return df.sort_values("zone_id").reset_index(drop=True)


def get_risk_map(episode_id: int, target_time: pd.Timestamp | None = None) -> gpd.GeoDataFrame:
    """Zonas en WGS84 unidas a su predicción, lista para pintar."""
    columns = ["zone_id", "name", "district", "neighbourhood", "altitude_avg", "slope_avg", "imperviousness",
               "patricova_level", "station_id", "station_dist_m", "geometry"]
    return get_zones_web()[columns].merge(get_predictions(episode_id, target_time), on="zone_id")


def get_district_summary(episode_id: int, target_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """Celdas por nivel y probabilidad media por distrito."""
    df = get_predictions(episode_id, target_time).merge(get_zones()[["zone_id", "district"]], on="zone_id")
    summary = df.pivot_table(index="district", columns="risk_level", values="zone_id", aggfunc="count", observed=False).fillna(0).astype(int)
    summary["probabilidad_media"] = df.groupby("district")["risk_probability"].mean().round(3)
    return summary.reset_index().sort_values(["Alto", "Medio"], ascending=False)


def get_zone_history(zone_id: int) -> pd.DataFrame:
    """Antecedentes de inundación de una celda: episodio, fechas, gravedad y fuente."""
    events = pd.read_parquet(FLOOD_EVENTS_FILE)
    events = events[events["zone_id"] == int(zone_id)]
    episodes = pd.read_parquet(EPISODES_FILE)[["episode_id", "name", "is_simulated"]]
    sources = get_data_sources()[["id", "name", "status"]].rename(columns={"id": "source_id", "name": "source"})
    return (
        events.merge(episodes, on="episode_id").merge(sources, on="source_id")
        [["name", "started_at", "ended_at", "severity", "source", "status"]]
        .rename(columns={"name": "episode"})
        .sort_values("started_at", ascending=False)
        .reset_index(drop=True)
    )


def get_district_history(district: str) -> pd.DataFrame:
    """Episodios con celdas inundadas en un distrito y fracción de celdas afectadas."""
    zones = get_zones()
    in_district = zones.loc[zones["district"] == district, "zone_id"]
    events = pd.read_parquet(FLOOD_EVENTS_FILE)
    events = events[events["zone_id"].isin(in_district)]
    episodes = pd.read_parquet(EPISODES_FILE)[["episode_id", "name", "started_at"]]
    summary = events.groupby("episode_id").size().rename("celdas_inundadas").reset_index().merge(episodes, on="episode_id")
    summary["fraccion_afectada"] = (summary["celdas_inundadas"] / len(in_district)).round(3)
    return summary.sort_values("started_at", ascending=False)[["name", "started_at", "celdas_inundadas", "fraccion_afectada"]]


def find_zone(lat: float, lon: float) -> int | None:
    """Celda que contiene un punto WGS84, o None si cae fuera del piloto."""
    point = gpd.GeoSeries([Point(lon, lat)], crs=CRS_WEB).to_crs(CRS).iloc[0]
    hit = get_zones()[get_zones().contains(point)]
    return None if hit.empty else int(hit["zone_id"].iloc[0])


def search_places(text: str) -> pd.DataFrame:
    """Busca barrios o distritos por nombre en los datos propios; devuelve nombre, tipo y centro WGS84."""
    zones = get_zones_web()
    text = text.strip().lower()
    rows = []
    for kind, column in (("Distrito", "district"), ("Barrio", "neighbourhood")):
        for name in zones[column].dropna().unique():
            if text and text in name.lower():
                centroid = zones[zones[column] == name].to_crs(CRS).union_all().centroid
                point = gpd.GeoSeries([centroid], crs=CRS).to_crs(CRS_WEB).iloc[0]
                rows.append({"name": name, "kind": kind, "lat": point.y, "lon": point.x})
    return pd.DataFrame(rows, columns=["name", "kind", "lat", "lon"]).drop_duplicates("name")
