"""Carga de cada fuente de data/preparados. Si falta un fichero, el pipeline se detiene: no hay datos simulados.

Ficheros esperados en data/preparados/ (nombres en config.PREPARED_FILES):
  mdt_altitud.tif                  MDT del CNIG/IGN (altitud en m)
  impermeabilidad_suelo.tif        Copernicus HRL Imperviousness Density (0-100 %)
  patricova_peligrosidad.gpkg      PATRICOVA peligrosidad, columna "nivel" (1-6)
  barrios_valencia.gpkg            Ayuntamiento de Valencia, columnas "nombre" y "distrito"
  zonas_inundadas_dana2024.gpkg    Copernicus EMS EMSR773 + huella GVA, columnas "episode", "severity" y "source"
  municipios.gpkg                  IGN/CNIG límites municipales, columna "municipio"
  cauces_principales.gpkg          IGN hidrografía: cauces principales, columna "nombre"
  altura_sobre_cauce.tif           altura sobre el cauce (HAND, m), calculada en la ingesta con el MDT y los cauces
  calados_campo_dana2024.gpkg      GVA DANA 2024: calado medido en campo, columna "depth_m" (no se usa para entrenar)
  suelo_no_urbano_clc2018.gpkg     Corine Land Cover 2018, arrozales y agua permanente (esas celdas se excluyen)
  geologia_igme.gpkg               IGME GEODE 50, unidades cuaternarias agrupadas, columna "grupo"
  snczi_zona_inundable_t500.gpkg   SNCZI zona inundable T=500 años (solo referencia)
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import box

from src.config import (
    CELL_SIZE_M, CITY_MUNICIPALITY, CRS, EPISODE_NAME, GEOLOGY_GROUPS, PREPARED_DIR, PREPARED_FILES, SEA_ALTITUDE_M,
    SEVERITY_DEPTH_LIMITS, SEVERITY_ORDER,
)

M_PER_KM = 1000


@dataclass
class SourceInfo:
    """Procedencia de una fuente: organismo, conjunto de datos, URL de descarga y licencia."""
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
        "Ayuntamiento de Valencia", "Barrios y distritos (geoportal, capas 224 y 225)",
        "https://geoportal.valencia.es/apps/OpenData/UrbanismoEInfraestructuras/", "CC BY 4.0",
    ),
    "flood_events": SourceInfo(
        "Copernicus EMS y GVA", "EMSR773 zonas inundadas (satélite) + huella oficial de la DANA 29-10-2024 (GVA / ICV)",
        "https://rapidmapping.emergency.copernicus.eu/EMSR773 ; https://terramapas.icv.gva.es/00_DANA2024",
        "Copernicus data policy (libre con atribución); GVA: por verificar",
    ),
    "municipalities": SourceInfo(
        "IGN/CNIG", "Límites municipales (recintos municipales)", "https://centrodedescargas.cnig.es", "CC BY 4.0",
    ),
    "channels": SourceInfo(
        "IGN", "Hidrografía de España (INSPIRE HY): cauces", "https://api-features.idee.es/collections/watercourse",
        "CC BY 4.0",
    ),
    "field_depths": SourceInfo(
        "GVA / Institut Cartogràfic Valencià", "DANA 2024: altura de agua medida en campo (IVE, SGRT, UPV)",
        "https://terramapas.icv.gva.es/00_DANA2024", "Por verificar",
    ),
    "non_urban": SourceInfo(
        "Copernicus Land (EEA)", "Corine Land Cover 2018: arrozales y agua permanente, para excluirlos",
        "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/0",
        "Copernicus data policy (libre con atribución)",
    ),
    "geology": SourceInfo(
        "IGME", "Mapa Geológico de España continuo 1:50.000 (GEODE 50): unidades cuaternarias",
        "https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geode_50/MapServer", "Por verificar",
    ),
    "snczi": SourceInfo(
        "MITECO (SNCZI)", "Zonas inundables T=500 años (solo referencia)",
        "https://www.miteco.gob.es/es/cartografia-y-sig/ide/descargas/agua/zi-lamina.html", "Por verificar",
    ),
}


def _path(key: str) -> Path:
    """Ruta del fichero preparado de la fuente; error si falta (se genera con la ingesta)."""
    path = PREPARED_DIR / PREPARED_FILES[key]
    if not path.exists():
        origin = SOURCES[key].name if key in SOURCES else "derivado de otras fuentes"
        raise FileNotFoundError(f"Falta {path} ({origin}); se genera con python -m src.pipeline.ingest")
    return path


def _raster_points(path, zones: gpd.GeoDataFrame, samples_per_side: int = 5) -> np.ndarray:
    """Valor del ráster en una rejilla de puntos interior a cada celda: matriz (celdas, puntos), nodata = NaN.

    Aplica el factor de escala del fichero (el MDT preparado guarda centímetros enteros)."""
    offsets = (np.arange(samples_per_side) + 0.5) / samples_per_side * CELL_SIZE_M
    bounds = zones.geometry.bounds
    xs = (bounds["minx"].to_numpy()[:, None, None] + offsets[None, :, None]).repeat(samples_per_side, 2)
    ys = (bounds["miny"].to_numpy()[:, None, None] + offsets[None, None, :]).repeat(samples_per_side, 1)
    with rasterio.open(path) as src:
        points = gpd.GeoSeries(gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=CRS).to_crs(src.crs)
        values = np.array([v[0] for v in src.sample(zip(points.x, points.y, strict=True))], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
        values = values * src.scales[0] + src.offsets[0]
    return values.reshape(len(zones), -1)


def _mean(values: np.ndarray) -> np.ndarray:
    """Media por celda de los puntos válidos; NaN si no queda ninguno."""
    valid = ~np.isnan(values)
    return np.where(valid.any(axis=1), np.nansum(values, axis=1) / np.maximum(valid.sum(axis=1), 1), np.nan)


def load_altitude(zones: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (altitud media de los puntos de tierra, fracción de puntos en el mar)."""
    values = _raster_points(_path("terrain"), zones)
    sea = values == SEA_ALTITUDE_M
    values[sea] = np.nan
    return _mean(values), sea.mean(axis=1)


def load_imperviousness(zones: gpd.GeoDataFrame) -> np.ndarray:
    """% de suelo sellado medio de cada celda (0-100); los puntos de mar no cuentan en la media."""
    values = _raster_points(_path("imperviousness"), zones)
    sea = _raster_points(_path("terrain"), zones) == SEA_ALTITUDE_M
    values[sea] = np.nan
    return _mean(values)


def load_patricova(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Nivel PATRICOVA más peligroso (1 = máximo) que toca cada celda; 0 si no toca ninguna zona."""
    hazard = gpd.read_file(_path("patricova")).to_crs(CRS)[["nivel", "geometry"]]
    joined = gpd.sjoin(zones[["zone_id", "geometry"]], hazard, how="left", predicate="intersects")
    level = joined.groupby("zone_id")["nivel"].min().reindex(zones["zone_id"]).fillna(0)
    return level.astype(int).to_numpy()


def _covered_fraction(zones: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, by: str | None = None) -> pd.DataFrame:
    """Fracción de cada celda cubierta por los polígonos (por grupo si se indica `by`).

    Cruce espacial con índice: cada celda solo se compara con los polígonos que tiene cerca. Los polígonos de un mismo
    grupo no deben solaparse: el solape se contaría dos veces.
    """
    columns = [by] if by else []
    polygons = polygons[[*columns, "geometry"]].explode(index_parts=False)
    pieces = gpd.overlay(zones[["zone_id", "geometry"]], polygons, how="intersection", keep_geom_type=True)
    pieces["_fraction"] = pieces.area / CELL_SIZE_M**2
    if by:
        table = pieces.pivot_table(index="zone_id", columns=by, values="_fraction", aggfunc="sum")
    else:
        table = pieces.groupby("zone_id")[["_fraction"]].sum()
    return table.reindex(zones["zone_id"]).fillna(0).clip(upper=1).set_axis(zones.index)


def load_geology(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Fracción de cada celda en cada grupo geológico (config.GEOLOGY_GROUPS), de 0 a 1."""
    geology = gpd.read_file(_path("geology")).to_crs(CRS)
    fractions = _covered_fraction(zones, geology, by="grupo").reindex(columns=list(GEOLOGY_GROUPS), fill_value=0)
    return fractions.round(3).rename_axis(columns=None)


def _touches(zones: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> np.ndarray:
    """1 si la celda toca alguno de los polígonos, 0 si no.

    Usa geometría preparada: pensado para pocos polígonos enormes, como las zonas del SNCZI."""
    cells, hit = zones.geometry.values, np.zeros(len(zones), dtype=bool)
    for polygon in polygons.geometry.values:
        shapely.prepare(polygon)
        hit |= shapely.intersects(polygon, cells)
    return hit.astype(int)


def load_snczi(zones: gpd.GeoDataFrame) -> np.ndarray:
    """1 si la celda toca la zona inundable del SNCZI (mismo criterio que PATRICOVA)."""
    return _touches(zones, gpd.read_file(_path("snczi")).to_crs(CRS))


def _largest_overlap(zones: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, columns: list[str]) -> pd.DataFrame:
    """Atributos del polígono que cubre más superficie de cada celda (NaN si no toca ninguno)."""
    pieces = gpd.overlay(zones[["zone_id", "geometry"]], polygons[[*columns, "geometry"]], how="intersection")
    pieces["_area"] = pieces.area
    best = pieces.sort_values("_area", ascending=False).drop_duplicates("zone_id").set_index("zone_id")
    return best[columns].reindex(zones["zone_id"])


def load_admin_units(zones: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (municipality, district, neighbourhood) por celda, según la mayor superficie compartida.

    Barrio y distrito solo existen en el municipio de Valencia; en el resto de municipios quedan vacíos.
    """
    municipalities = gpd.read_file(_path("municipalities")).to_crs(CRS)
    barrios = gpd.read_file(_path("neighbourhoods")).to_crs(CRS)
    municipality = _largest_overlap(zones, municipalities, ["municipio"])["municipio"]
    # Celdas de playa (Pinedo, El Saler): los recintos municipales siguen la línea de costa y no las tocan
    missing = municipality.isna().to_numpy()
    if missing.any():
        beach = zones.loc[missing, ["zone_id"]].set_geometry(zones.geometry[missing].centroid)
        nearest = gpd.sjoin_nearest(beach, municipalities[["municipio", "geometry"]]).drop_duplicates("zone_id")
        municipality.loc[nearest["zone_id"].to_numpy()] = nearest["municipio"].to_numpy()
    hood = _largest_overlap(zones, barrios, ["nombre", "distrito"])
    in_city = (municipality == CITY_MUNICIPALITY).to_numpy()
    return municipality.to_numpy(), hood["distrito"].where(in_city).to_numpy(), hood["nombre"].where(in_city).to_numpy()


def channel_distance(zones: gpd.GeoDataFrame, channels: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distancia (km) del centro de cada celda al cauce principal más cercano y nombre de ese cauce."""
    centroids = zones[["zone_id"]].set_geometry(zones.geometry.centroid)
    nearest = gpd.sjoin_nearest(centroids, channels[["nombre", "geometry"]], how="left", distance_col="_dist")
    nearest = nearest.drop_duplicates("zone_id").set_index("zone_id").reindex(zones["zone_id"])
    return (nearest["_dist"] / M_PER_KM).to_numpy(), nearest["nombre"].to_numpy()


def load_channel_distance(zones: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    return channel_distance(zones, gpd.read_file(_path("channels")).to_crs(CRS))


def load_height_above_channel(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Altura media (m) de cada celda sobre el cauce o el mar al que llega su agua siguiendo la pendiente (HAND, Height
    Above Nearest Drainage); la calcula la ingesta. Los puntos de mar no cuentan en la media."""
    values = _raster_points(_path("hand"), zones)
    sea = _raster_points(_path("terrain"), zones) == SEA_ALTITUDE_M
    values[sea] = np.nan
    return _mean(values)


def load_non_urban_fraction(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Fracción de cada celda ocupada por arrozal o agua permanente (Corine Land Cover 2018)."""
    cover = gpd.read_file(_path("non_urban")).to_crs(CRS)
    return _covered_fraction(zones, cover)["_fraction"].to_numpy()


def load_field_depths() -> gpd.GeoDataFrame:
    """Puntos de calado medido en campo (nunca entran en el entrenamiento como variable)."""
    return gpd.read_file(_path("field_depths")).to_crs(CRS)


def depth_severity(depth_m: pd.Series) -> pd.Series:
    """Gravedad a partir del calado, con los mismos cortes que Copernicus EMS."""
    moderate, severe = SEVERITY_DEPTH_LIMITS
    return pd.Series(np.select([depth_m > severe, depth_m >= moderate], ["Grave", "Moderada"], default="Leve"),
                     index=depth_m.index)


def load_observed_floods(zones: gpd.GeoDataFrame) -> pd.DataFrame:
    """Por celda: fracción cubierta por agua en la DANA, severidad máxima y fuentes que la marcan
    (zone_id, flooded_fraction, severity, source)."""
    floods = gpd.read_file(_path("flood_events")).to_crs(CRS)
    cells = zones[["zone_id", "geometry"]]
    # Se recorta al rectángulo de las celdas antes de disolver: las capas de inundación son pocos polígonos enormes
    floods = floods[floods["episode"] == EPISODE_NAME].clip(box(*cells.total_bounds))

    # Fracción de la celda bajo cualquier polígono inundado (se disuelven para no contar solapes dos veces)
    water = gpd.overlay(cells, floods[["geometry"]].dissolve(), how="intersection")
    fraction = (water.set_index("zone_id").area / CELL_SIZE_M**2).clip(upper=1)

    # Severidad más grave entre los polígonos que tocan la celda y fuentes que la marcan
    rank = floods.assign(rank=floods["severity"].map(SEVERITY_ORDER.index))
    joined = gpd.sjoin(cells, rank[["rank", "source", "geometry"]], predicate="intersects")
    severity_rank = joined.groupby("zone_id")["rank"].max()
    source = joined.groupby("zone_id")["source"].agg(lambda s: " y ".join(sorted(set(s))))

    # La huella de la GVA no trae calado: si en la celda se midió más agua en campo, manda el calado medido
    points = gpd.sjoin(load_field_depths(), cells, predicate="within")
    measured = depth_severity(points["depth_m"]).map(SEVERITY_ORDER.index).groupby(points["zone_id"]).max()
    severity_rank = pd.concat([severity_rank, measured.reindex(severity_rank.index)], axis=1).max(axis=1)

    result = pd.DataFrame({"zone_id": zones["zone_id"]})
    result["flooded_fraction"] = result["zone_id"].map(fraction).fillna(0).round(4).to_numpy()
    result["severity"] = result["zone_id"].map(severity_rank.astype(int).map(lambda r: SEVERITY_ORDER[r])).to_numpy()
    result["source"] = result["zone_id"].map(source).to_numpy()
    return result


def data_sources_table(updated_at: pd.Timestamp) -> pd.DataFrame:
    """Tabla DataSource: procedencia y licencia de cada fuente usada en esta ejecución."""
    rows = [
        {
            "id": i + 1,
            "key": key,
            "name": info.name,
            "dataset_name": info.dataset_name,
            "url": info.url,
            "license": info.license,
            "file": PREPARED_FILES[key],
            "updated_at": updated_at,
        }
        for i, (key, info) in enumerate(SOURCES.items())
    ]
    return pd.DataFrame(rows)
