"""Comprobaciones de datos sobre los Parquet procesados; sustituyen a las claves foráneas."""

import geopandas as gpd
import pandas as pd

from src.config import CITY_MUNICIPALITY, CRS, GEOLOGY_GROUPS, N_FOLDS, PATRICOVA_MAX_LEVEL, SNCZI_COLUMN


def check_zones(zones: gpd.GeoDataFrame) -> list[str]:
    """Comprueba las zonas (CRS, identificadores, rangos, municipio y barrio); devuelve los errores."""
    errors = []
    if zones.crs is None or zones.crs.to_string() != CRS:
        errors.append(f"zones: CRS {zones.crs} distinto de {CRS}")
    if zones["zone_id"].duplicated().any():
        errors.append("zones: zone_id duplicado")
    if not zones["imperviousness"].between(0, 100).all():
        errors.append("zones: imperviousness fuera de 0-100")
    if not zones["patricova_level"].between(0, PATRICOVA_MAX_LEVEL).all():
        errors.append(f"zones: patricova_level fuera de 0-{PATRICOVA_MAX_LEVEL}")
    if zones[["altitude_avg", "slope_avg"]].isna().any().any():
        errors.append("zones: altitud o pendiente sin valor")
    if zones["municipality"].isna().any():
        errors.append("zones: celda sin municipio")
    if (zones["dist_channel_km"] < 0).any() or zones["dist_channel_km"].isna().any():
        errors.append("zones: distancia al cauce negativa o vacía")
    if zones.loc[zones["municipality"] != CITY_MUNICIPALITY, "neighbourhood"].notna().any():
        errors.append("zones: barrio de Valencia asignado a una celda de otro municipio")
    if not zones[list(GEOLOGY_GROUPS)].stack().between(0, 1).all():
        errors.append("zones: fracción geológica fuera de 0-1")
    if not zones[SNCZI_COLUMN].isin([0, 1]).all():
        errors.append("zones: zona SNCZI distinta de 0/1")
    return errors


def check_flood_events(events: pd.DataFrame, zones: gpd.GeoDataFrame) -> list[str]:
    """Comprueba las celdas inundadas (existen en zones, fracción 0-1, gravedad y fuente); devuelve los errores."""
    errors = []
    if not events["zone_id"].isin(zones["zone_id"]).all():
        errors.append("flood_events: zone_id inexistente en zones")
    if not events["flooded_fraction"].between(0, 1).all():
        errors.append("flood_events: flooded_fraction fuera de 0-1")
    if events["severity"].isna().any():
        errors.append("flood_events: celda inundada sin severidad")
    if events["source"].isna().any():
        errors.append("flood_events: celda inundada sin fuente (Copernicus EMS o GVA)")
    return errors


def check_training(table: pd.DataFrame, features: list[str]) -> list[str]:
    """Comprueba la tabla de entrenamiento (sin huecos, etiqueta 0/1, cada bloque en un solo grupo y las dos clases
    en cada grupo); devuelve los errores."""
    errors = []
    if table[features].isna().any().any():
        errors.append("training: variables con huecos")
    if not table["flooded"].isin([0, 1]).all():
        errors.append("training: etiqueta distinta de 0/1")
    if table.groupby("block_id")["fold"].nunique().gt(1).any():
        errors.append("training: un bloque espacial repartido entre varios grupos")
    for fold in range(N_FOLDS):
        if table.loc[table["fold"] == fold, "flooded"].nunique() < 2:
            errors.append(f"training: el grupo {fold} no tiene las dos clases")
    return errors
