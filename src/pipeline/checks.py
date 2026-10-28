"""Comprobaciones de datos sobre los Parquet procesados; sustituyen a las claves foráneas."""

import geopandas as gpd
import pandas as pd

from src.config import CRS


def check_zones(zones: gpd.GeoDataFrame) -> list[str]:
    errors = []
    if zones.crs is None or zones.crs.to_string() != CRS:
        errors.append(f"zones: CRS {zones.crs} distinto de {CRS}")
    if zones["zone_id"].duplicated().any():
        errors.append("zones: zone_id duplicado")
    if not zones["imperviousness"].between(0, 100).all():
        errors.append("zones: imperviousness fuera de 0-100")
    if not zones["patricova_level"].between(0, 6).all():
        errors.append("zones: patricova_level fuera de 0-6")
    if zones[["altitude_avg", "slope_avg"]].isna().any().any():
        errors.append("zones: altitud o pendiente sin valor")
    return errors


def check_weather(weather: pd.DataFrame, zones: gpd.GeoDataFrame, episodes: pd.DataFrame) -> list[str]:
    errors = []
    if (weather[["precipitation_mm", "precipitation_1h", "precipitation_24h"]] < 0).any().any():
        errors.append("weather: lluvia negativa")
    if (weather["precipitation_24h"] + 1e-6 < weather["precipitation_1h"]).any():
        errors.append("weather: acumulado 24 h menor que el de 1 h")
    if not weather["zone_id"].isin(zones["zone_id"]).all():
        errors.append("weather: zone_id inexistente en zones")
    if not weather["episode_id"].isin(episodes["episode_id"]).all():
        errors.append("weather: episode_id inexistente en episodes")
    if weather.duplicated(["zone_id", "episode_id", "observed_at"]).any():
        errors.append("weather: observaciones duplicadas")
    return errors


def check_flood_events(events: pd.DataFrame, zones: gpd.GeoDataFrame, episodes: pd.DataFrame) -> list[str]:
    errors = []
    if not events["zone_id"].isin(zones["zone_id"]).all():
        errors.append("flood_events: zone_id inexistente en zones")
    scenario_ids = episodes.loc[episodes["kind"] != "historical", "episode_id"]
    if events["episode_id"].isin(scenario_ids).any():
        errors.append("flood_events: hay inundaciones asociadas a escenarios o previsiones")
    if (events["ended_at"] < events["started_at"]).any():
        errors.append("flood_events: ended_at anterior a started_at")
    return errors


def check_training(table: pd.DataFrame, features: list[str]) -> list[str]:
    errors = []
    if table[features].isna().any().any():
        errors.append("training: variables con huecos")
    if not table["flooded"].isin([0, 1]).all():
        errors.append("training: etiqueta distinta de 0/1")
    if table["flooded"].nunique() < 2:
        errors.append("training: solo hay una clase")
    return errors
