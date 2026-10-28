"""Carga de cada fuente: usa el fichero real de data/raw si existe y, si no, la muestra simulada.

Ficheros reales esperados en data/raw/ (se dejan tal cual se descargan):
  mdt.tif                 MDT del CNIG/IGN (altitud en m)
  imperviousness.tif      Copernicus HRL Imperviousness Density (0-100 %)
  patricova.gpkg          PATRICOVA peligrosidad, columna "nivel" (1-6)
  barrios.geojson         Ayuntamiento de Valencia, columnas "nombre" y "distrito"
  stations.csv            station_id, name, lat, lon
  episodes.csv            name, started_at, ended_at, kind
  aemet_observations.csv  station_id, observed_at, precipitation_mm, temperature_c
  flood_events.gpkg       Copernicus EMS, columnas "episode" y "severity"
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.config import CELL_SIZE_M, CRS, CRS_WEB, RAW_DIR
from src.pipeline import sample


@dataclass
class SourceInfo:
    name: str
    dataset_name: str
    url: str
    license: str


SOURCES = {
    "terrain": SourceInfo("CNIG/IGN", "MDT05 modelo digital del terreno", "https://centrodedescargas.cnig.es", "CC BY 4.0"),
    "imperviousness": SourceInfo(
        "Copernicus Land", "HRL Imperviousness Density",
        "https://land.copernicus.eu/en/products/high-resolution-layer-imperviousness",
        "Copernicus data policy (libre con atribución)",
    ),
    "patricova": SourceInfo(
        "PATRICOVA (GVA)", "Peligrosidad de inundación",
        "https://politicaterritorial.gva.es/es/web/planificacion-territorial-e-infraestructura-verde/patricova",
        "Por verificar",
    ),
    "neighbourhoods": SourceInfo(
        "Ayuntamiento de Valencia", "Barrios y distritos", "https://valencia.opendatasoft.com", "Por verificar"
    ),
    "weather": SourceInfo("AEMET OpenData", "Observaciones y previsión horaria", "https://opendata.aemet.es", "Aviso legal AEMET (reutilización con atribución)"),
    "flood_events": SourceInfo(
        "Copernicus EMS", "Delimitación de zonas inundadas", "https://emergency.copernicus.eu",
        "Copernicus data policy (libre con atribución)",
    ),
}

# Estado de cada fuente en esta ejecución ("real" o "simulado"), lo rellenan los loaders
status: dict[str, str] = {}


def _raster_mean(path, zones: gpd.GeoDataFrame, samples_per_side: int = 5) -> np.ndarray:
    """Media del ráster en una rejilla de puntos interior a cada celda."""
    offsets = (np.arange(samples_per_side) + 0.5) / samples_per_side * CELL_SIZE_M
    bounds = zones.geometry.bounds
    xs = (bounds["minx"].to_numpy()[:, None, None] + offsets[None, :, None]).repeat(samples_per_side, 2)
    ys = (bounds["miny"].to_numpy()[:, None, None] + offsets[None, None, :]).repeat(samples_per_side, 1)
    with rasterio.open(path) as src:
        points = gpd.GeoSeries(gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=CRS).to_crs(src.crs)
        values = np.array([v[0] for v in src.sample(zip(points.x, points.y))], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
    return np.nanmean(values.reshape(len(zones), -1), axis=1)


def load_altitude(zones: gpd.GeoDataFrame) -> np.ndarray:
    path = RAW_DIR / "mdt.tif"
    status["terrain"] = "real" if path.exists() else "simulado"
    return _raster_mean(path, zones) if path.exists() else sample.terrain(zones)


def load_imperviousness(zones: gpd.GeoDataFrame) -> np.ndarray:
    path = RAW_DIR / "imperviousness.tif"
    status["imperviousness"] = "real" if path.exists() else "simulado"
    return _raster_mean(path, zones) if path.exists() else sample.imperviousness(zones)


def load_patricova(zones: gpd.GeoDataFrame, altitude: np.ndarray) -> np.ndarray:
    path = RAW_DIR / "patricova.gpkg"
    status["patricova"] = "real" if path.exists() else "simulado"
    if not path.exists():
        return sample.patricova(zones, altitude)
    hazard = gpd.read_file(path).to_crs(CRS)[["nivel", "geometry"]]
    # Nivel más peligroso (número más bajo) que toca la celda; 0 si ninguno
    joined = gpd.sjoin(zones[["zone_id", "geometry"]], hazard, how="left", predicate="intersects")
    level = joined.groupby("zone_id")["nivel"].min().reindex(zones["zone_id"]).fillna(0)
    return level.astype(int).to_numpy()


def load_neighbourhoods(zones: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (district, neighbourhood) por celda."""
    path = RAW_DIR / "barrios.geojson"
    status["neighbourhoods"] = "real" if path.exists() else "simulado"
    if not path.exists():
        district = sample.districts(zones)
        return district, district
    barrios = gpd.read_file(path).to_crs(CRS)[["nombre", "distrito", "geometry"]]
    centroids = zones[["zone_id"]].set_geometry(zones.geometry.centroid)
    joined = gpd.sjoin_nearest(centroids, barrios, how="left").drop_duplicates("zone_id")
    return joined["distrito"].to_numpy(), joined["nombre"].to_numpy()


def load_stations() -> gpd.GeoDataFrame:
    path = RAW_DIR / "stations.csv"
    if not path.exists():
        return sample.stations()
    df = pd.read_csv(path, dtype={"station_id": str})
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WEB).to_crs(CRS)


def load_episodes() -> pd.DataFrame:
    path = RAW_DIR / "episodes.csv"
    if not path.exists():
        return sample.episodes()
    df = pd.read_csv(path, parse_dates=["started_at", "ended_at"])
    df.insert(0, "episode_id", range(1, len(df) + 1))
    df["is_simulated"] = df["kind"].eq("scenario")
    return df


def load_station_rain(stations: gpd.GeoDataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Lluvia horaria por estación con su episodio asignado."""
    path = RAW_DIR / "aemet_observations.csv"
    status["weather"] = "real" if path.exists() else "simulado"
    if not path.exists():
        return sample.station_rain(stations, episodes)
    obs = pd.read_csv(path, dtype={"station_id": str}, parse_dates=["observed_at"])
    parts = []
    for _, ep in episodes.iterrows():
        window = obs[obs["observed_at"].between(ep["started_at"], ep["ended_at"])].copy()
        window["episode_id"] = ep["episode_id"]
        parts.append(window)
    return pd.concat(parts, ignore_index=True)


def load_observed_floods(zones: gpd.GeoDataFrame, episodes: pd.DataFrame) -> pd.DataFrame | None:
    """Celdas inundadas por episodio (zone_id, episode_id, severity) o None si no hay fichero real."""
    path = RAW_DIR / "flood_events.gpkg"
    status["flood_events"] = "real" if path.exists() else "simulado"
    if not path.exists():
        return None
    floods = gpd.read_file(path).to_crs(CRS)[["episode", "severity", "geometry"]]
    floods = floods.merge(episodes[["episode_id", "name"]], left_on="episode", right_on="name")
    joined = gpd.sjoin(zones[["zone_id", "geometry"]], floods, predicate="intersects")
    return joined[["zone_id", "episode_id", "severity"]].drop_duplicates(["zone_id", "episode_id"])


def data_sources_table(updated_at: pd.Timestamp) -> pd.DataFrame:
    """Tabla DataSource con el estado real/simulado de cada fuente en esta ejecución."""
    rows = [
        {
            "id": i + 1,
            "key": key,
            "name": info.name,
            "dataset_name": info.dataset_name,
            "url": info.url,
            "license": info.license,
            "status": status.get(key, "simulado"),
            "updated_at": updated_at,
        }
        for i, (key, info) in enumerate(SOURCES.items())
    ]
    return pd.DataFrame(rows)
