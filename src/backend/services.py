"""Backend: lectura de los resultados del pipeline (Parquet) para el frontend.

El frontend solo usa estas funciones; si se migra a una base de datos o a una API, solo cambia este módulo.
La búsqueda de direcciones la hace el navegador contra CartoCiudad (IGN): la aplicación no guarda lo que se busca
(Fase 1, 9.1).
"""

import base64
import json
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS as ProjCRS
from shapely.geometry import Point

from src.config import (
    CELL_SIZE_M, CRS, CRS_WEB, DATA_SOURCES_FILE, EPISODE_NAME, FLOOD_EVENTS_FILE, MODEL_CARD_FILE, PILOT_EXTENT,
    PREDICTIONS_DIR, RISK_LEVELS, SEVERITY_ORDER, SNCZI_COLUMN, ZONES_FILE,
)
from src.pipeline.features import area_unit

# Cuadrícula del mapa: un byte por celda. Bits 0-1: nivel de riesgo (0 sin celda, 1 Bajo, 2 Medio, 3 Alto);
# bits 2-3: gravedad observada en la DANA (0 no inundada, 1 Leve, 2 Moderada, 3 Grave)
SEVERITY_SHIFT = 2


@lru_cache
def get_zones() -> gpd.GeoDataFrame:
    """Zonas en EPSG:25830 con sus atributos físicos, distrito y barrio."""
    return gpd.read_parquet(ZONES_FILE)


def get_zones_web() -> gpd.GeoDataFrame:
    return get_zones().to_crs(CRS_WEB)


@lru_cache
def get_model_card() -> dict:
    """Ficha del modelo (data/modelos/model_card.json) como diccionario."""
    return json.loads(MODEL_CARD_FILE.read_text(encoding="utf-8"))


def get_data_sources() -> pd.DataFrame:
    """Fuentes de datos de la última ejecución del pipeline: procedencia, licencia y fichero."""
    return pd.read_csv(DATA_SOURCES_FILE)


def get_predictions(model_version: str | None = None) -> pd.DataFrame:
    """Predicción por celda del modelo indicado (por defecto, el de la ficha)."""
    version = model_version or get_model_card()["model_version"]
    df = pd.read_parquet(PREDICTIONS_DIR / f"model_version={version}").assign(model_version=version)
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=RISK_LEVELS, ordered=True)
    return df.sort_values("zone_id").reset_index(drop=True)


def get_risk_map() -> gpd.GeoDataFrame:
    """Zonas en WGS84 unidas a su predicción y a lo observado en la DANA, lista para pintar."""
    columns = ["zone_id", "name", "municipality", "district", "neighbourhood", "altitude_avg", "slope_avg",
               "imperviousness", "geology", "patricova_level", SNCZI_COLUMN, "dist_channel_km", "channel_name", "geometry"]
    events = pd.read_parquet(FLOOD_EVENTS_FILE)[["zone_id", "flooded_fraction", "severity"]]
    data = get_zones_web()[columns].merge(get_predictions(), on="zone_id").merge(events, on="zone_id", how="left")
    data["observed"] = data["severity"].fillna("No inundada")
    data["place"] = area_unit(data, "barrio")
    # Posición relativa: % de celdas del piloto con menor susceptibilidad que esta
    data["risk_percentile"] = (100 * data["risk_probability"].rank(pct=True)).round().astype(int)
    return data


def get_district_summary() -> pd.DataFrame:
    """Celdas por nivel, probabilidad media y celdas inundadas en la DANA por distrito (municipio fuera de València)."""
    zones = get_zones()
    df = get_predictions().merge(zones[["zone_id"]].assign(zona=area_unit(zones, "distrito").to_numpy()), on="zone_id")
    summary = df.pivot_table(index="zona", columns="risk_level", values="zone_id", aggfunc="count", observed=False)
    summary = summary.fillna(0).astype(int)
    summary["probabilidad_media"] = df.groupby("zona")["risk_probability"].mean().round(3)
    summary["inundadas_dana"] = df.groupby("zona")["flooded"].sum().astype(int)
    return summary.reset_index().sort_values(["Alto", "Medio"], ascending=False)


@lru_cache
def get_map_grid() -> dict:
    """Cuadrícula del área piloto para el mapa del navegador, que construye ahí las celdas (un byte por celda).

    Devuelve el origen (esquina suroeste, en EPSG:25830), el tamaño de celda, filas, columnas, el huso UTM y los códigos
    de cada celda (ver SEVERITY_SHIFT) en base64, fila a fila desde el sur. Las celdas se numeran como en
    grid.build_grid: zone_id = row * cols + col + 1.
    """
    x0, y0, x1, y1 = PILOT_EXTENT
    cols, rows = len(np.arange(x0, x1, CELL_SIZE_M)), len(np.arange(y0, y1, CELL_SIZE_M))
    zones = get_zones()
    if not (zones["zone_id"] == zones["row"] * cols + zones["col"] + 1).all():
        raise ValueError("zone_id no sigue la numeración de la cuadrícula")
    utm_zone = ProjCRS(CRS).utm_zone
    if not utm_zone or not utm_zone.endswith("N"):
        raise ValueError(f"{CRS} no es una proyección UTM del hemisferio norte")

    levels = get_predictions().set_index("zone_id")["risk_level"].cat.codes + 1
    events = pd.read_parquet(FLOOD_EVENTS_FILE).set_index("zone_id")["severity"]
    severity = events.map(SEVERITY_ORDER.index).add(1).reindex(zones["zone_id"]).fillna(0).astype(np.uint8)
    codes = np.zeros(rows * cols, dtype=np.uint8)
    cell = (zones["row"] * cols + zones["col"]).to_numpy()
    codes[cell] = levels.reindex(zones["zone_id"]).to_numpy().astype(np.uint8) | (severity.to_numpy() << SEVERITY_SHIFT)
    return {"origin": [float(x0), float(y0)], "cell_m": CELL_SIZE_M, "rows": rows, "cols": cols,
            "utm_zone": int(utm_zone[:-1]), "codes": base64.b64encode(codes.tobytes()).decode("ascii")}


def get_zone_history(zone_id: int) -> pd.DataFrame:
    """Lo observado en la celda durante la DANA: severidad, fracción inundada y fuente (Copernicus EMS y/o GVA)."""
    events = pd.read_parquet(FLOOD_EVENTS_FILE)
    events = events[events["zone_id"] == int(zone_id)]
    return events.assign(episode=EPISODE_NAME)[["episode", "severity", "flooded_fraction", "source"]].reset_index(drop=True)


def find_zone(lat: float, lon: float) -> int | None:
    """Celda que contiene un punto WGS84, o None si cae fuera del piloto."""
    point = gpd.GeoSeries([Point(lon, lat)], crs=CRS_WEB).to_crs(CRS).iloc[0]
    hit = get_zones()[get_zones().contains(point)]
    return None if hit.empty else int(hit["zone_id"].iloc[0])


@lru_cache
def list_places() -> pd.DataFrame:
    """Municipios, distritos y barrios del área piloto con su centro en WGS84 (name, kind, lat, lon), para las
    sugerencias del buscador. Las celdas miden lo mismo: el centro de un lugar es la media de los centros de sus
    celdas."""
    zones = get_zones()
    centers = zones.geometry.centroid
    frames = []
    for kind, column in (("Municipio", "municipality"), ("Distrito", "district"), ("Barrio", "neighbourhood")):
        mean = pd.DataFrame({"name": zones[column], "x": centers.x, "y": centers.y}).dropna().groupby("name").mean()
        points = gpd.GeoSeries(gpd.points_from_xy(mean["x"], mean["y"]), crs=CRS).to_crs(CRS_WEB)
        frames.append(pd.DataFrame({"name": mean.index, "kind": kind, "lat": points.y.to_numpy(),
                                    "lon": points.x.to_numpy()}))
    return pd.concat(frames, ignore_index=True).drop_duplicates("name").reset_index(drop=True)


def search_places(text: str) -> pd.DataFrame:
    """Municipios, distritos o barrios cuyo nombre contiene el texto (sin distinguir mayúsculas)."""
    text = text.strip().lower()
    places = list_places()
    return places[places["name"].str.lower().str.contains(text, regex=False)] if text else places.iloc[:0]
