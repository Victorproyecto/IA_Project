"""Ingesta: prepara data/preparados a partir de las descargas originales. Uso: python -m src.pipeline.ingest

Entrada (data/raw/, ver docs/DATOS.md):
  cnig_mdt05/PNOA_MDT05_*.tif                 hojas del MDT de 5 m del IGN
  copernicus_land_imd/CLMS_NVLCC_IMD_*.tif    Imperviousness Density 2024, 10 m
  gva_patricova/*.shp                         PATRICOVA peligrosidad de inundación
  ayuntamiento_valencia/barrios.geojson       barrios de Valencia
  ayuntamiento_valencia/distritos.geojson     distritos de Valencia
  copernicus_ems_EMSR773/*.shp                zonas inundadas y calado de la DANA
  cnig_limites_municipales/recintos_municipales_*.shp   límites municipales del IGN/CNIG
  ign_hidrografia/watercourse.geojson         cauces del IGN (Hidrografía INSPIRE)
  gva_dana2024/huella_inundacion.geojson      huella oficial de la zona inundada en la DANA (GVA / ICV)
  gva_dana2024/altura_agua_*.geojson          calados medidos en campo tras la DANA (IVE, SGRT, UPV)
  copernicus_clc/clc2018_no_urbano.geojson    arrozales y agua permanente de Corine Land Cover 2018
  igme_geode50/geode50.geojson                geología continua 1:50.000 del IGME (GEODE 50)
  snczi/laminasPB-q500.zip                    zona inundable del SNCZI (MITECO), T=500 años

Salida (data/preparados/): un fichero por fuente, con los nombres de config.PREPARED_FILES, más la altura sobre el
cauce (HAND), que se calcula a partir del MDT y los cauces ya preparados.

Todo se recorta al rectángulo que contiene el área piloto y la Ribera (prueba externa), con un margen.
"""

import logging
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pysheds.grid import Grid
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.validation import make_valid

from src.config import (
    CRS, RAW_DIR, EPISODE_NAME, EXCLUDED_CHANNEL_NAMES, EXTERNAL_TEST_EXTENT, GEOLOGY_GROUPS, MAIN_CHANNEL_NAMES,
    NON_URBAN_CLC_CODES, PATRICOVA_MAX_LEVEL, PILOT_EXTENT, PREPARED_DIR, PREPARED_FILES, SEA_ALTITUDE_M,
    SNCZI_RETURN_PERIOD,
)

log = logging.getLogger("ingest")

# El MDT preparado guarda la altitud en unidades de 1 cm
MDT_SCALE_M = 0.01
# Margen alrededor de la zona de datos para las capas vectoriales, para no cortar celdas del borde
MARGIN_M = 2000
# Margen mayor para los cauces: el cauce más cercano a una celda del borde puede quedar fuera de la zona
CHANNEL_MARGIN_M = 5000
# La huella de la GVA reparada puede diferir como mucho un 1 % de la superficie que declara la propia GVA
GVA_AREA_TOLERANCE = 0.01
M2_PER_HA = 10_000
# Altura sobre el cauce (HAND): se calcula sobre el MDT remuestreado a 10 m (de sobra para celdas de 100 m, y 4 veces
# menos píxeles que el de 5 m). Los cauces y el mar se rebajan HAND_BURN_M en el MDT para que el agua siga el trazado del
# IGN aunque el MDT no marque bien el fondo del cauce (puentes, encauzamientos)
HAND_RES_M = 10
HAND_BURN_M = 10
# Dirección del flujo D8 con la codificación de pysheds: código -> (fila, columna) del píxel vecino
D8_MOVES = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}
DEPTH_TO_SEVERITY = {
    "Below 0.50": "Leve",
    "0.50 - 1.00": "Moderada",
    "1.00 - 2.00": "Grave",
    "2.00 - 4.00": "Grave",
    "4.00 - 6.00": "Grave",
}


def _area(crs=CRS, margin_m: float = MARGIN_M) -> tuple[float, float, float, float]:
    """Rectángulo que contiene el área piloto y la prueba externa, con margen, en el CRS indicado."""
    x0, y0 = min(PILOT_EXTENT[0], EXTERNAL_TEST_EXTENT[0]), min(PILOT_EXTENT[1], EXTERNAL_TEST_EXTENT[1])
    x1, y1 = max(PILOT_EXTENT[2], EXTERNAL_TEST_EXTENT[2]), max(PILOT_EXTENT[3], EXTERNAL_TEST_EXTENT[3])
    return transform_bounds(CRS, crs, x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m)


def _files(folder: str, pattern: str) -> list[Path]:
    """Ficheros de data/raw/<folder> que cumplen el patrón, ordenados; error si no hay ninguno."""
    files = sorted((RAW_DIR / folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay {pattern} en {RAW_DIR / folder} (ver docs/DATOS.md)")
    return files


def _output(source: str) -> Path:
    """Ruta de data/preparados donde se guarda la fuente indicada."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    return PREPARED_DIR / PREPARED_FILES[source]


def build_mdt() -> None:
    """Une las hojas del MDT05, las recorta a la zona de datos y guarda la altitud en centímetros enteros.

    En centímetros enteros ocupa ~47 MB (146 MB en metros con decimales) sin perder nada por encima de 1 cm, muy por
    debajo de la precisión del propio MDT. El fichero lleva el factor de
    escala (0,01) y sources lo aplica al leer. El mar (0 exacto) se conserva: la tierra que redondearía a 0 cm se
    queda a ±1 cm. Sin margen: solo se muestrea dentro de las celdas.
    """
    sources = [rasterio.open(f) for f in _files("cnig_mdt05", "PNOA_MDT05_*.tif")]
    mosaic, transform = merge(sources, bounds=_area(sources[0].crs, margin_m=0))
    profile, nodata = sources[0].profile, sources[0].nodata
    for src in sources:
        src.close()
    valid = mosaic != nodata
    altitude_cm = np.round(mosaic / MDT_SCALE_M)
    near_sea_level = valid & (altitude_cm == 0) & (mosaic != SEA_ALTITUDE_M)
    altitude_cm[near_sea_level] = np.sign(mosaic[near_sea_level])
    altitude_cm[~valid] = nodata
    profile = profile | {
        "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform, "dtype": "int32", "nodata": nodata,
        "tiled": True, "compress": "zstd", "zstd_level": 22, "predictor": 2,
    }
    with rasterio.open(_output("terrain"), "w", **profile) as dst:
        dst.write(altitude_cm.astype("int32"))
        dst.scales = (MDT_SCALE_M,)


def build_imperviousness() -> None:
    """Recorta la tesela de Copernicus a la zona de datos (se mantiene su CRS, EPSG:3035).

    El valor 255 es mar o fuera de cobertura; la tesela no lo declara como nodata, así que se declara aquí.
    """
    with rasterio.open(_files("copernicus_land_imd", "CLMS_NVLCC_IMD_*.tif")[0]) as src:
        window = from_bounds(*_area(src.crs, margin_m=0), transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=window)
        profile = src.profile | {"height": data.shape[1], "width": data.shape[2],
                                 "transform": src.window_transform(window), "compress": "deflate", "nodata": 255}
    with rasterio.open(_output("imperviousness"), "w", **profile) as dst:
        dst.write(data)


def build_patricova() -> None:
    """Niveles 1-6 de PATRICOVA; el 7 (peligrosidad geomorfológica, sin frecuencia ni calado) se descarta."""
    hazard = gpd.read_file(_files("gva_patricova", "*.shp")[0]).to_crs(CRS)
    hazard = hazard[hazard["n_pelig"].between(1, PATRICOVA_MAX_LEVEL)].rename(columns={"n_pelig": "nivel"})
    hazard["nivel"] = hazard["nivel"].astype(int)
    hazard = hazard.clip(box(*_area()))[["codigo", "d_pelig", "nivel", "geometry"]]
    hazard.to_file(_output("patricova"), driver="GPKG")


def build_neighbourhoods() -> None:
    """Barrios con el nombre de su distrito."""
    folder = RAW_DIR / "ayuntamiento_valencia"
    barrios = gpd.read_file(folder / "barrios.geojson")
    # Un distrito puede venir partido en varios polígonos: basta un nombre por código
    distritos = (gpd.read_file(folder / "distritos.geojson")[["coddistrit", "nombre"]]
                 .drop_duplicates("coddistrit").rename(columns={"nombre": "distrito"}))
    barrios = barrios.merge(distritos, on="coddistrit")[["coddistbar", "coddistrit", "nombre", "distrito", "geometry"]]
    barrios.to_file(_output("neighbourhoods"), driver="GPKG")


def gva_footprint() -> gpd.GeoSeries:
    """Huella oficial de la zona inundada (GVA), con la geometría reparada.

    El servicio del ICV entrega la huella como un único polígono cuyos "huecos" son en realidad partes separadas de la
    zona inundada (geometría inválida, superficie negativa). make_valid reconstruye las partes; se comprueba que la
    superficie resultante coincide con la que declara la propia GVA (area_ha).
    """
    footprint = gpd.read_file(_files("gva_dana2024", "huella_inundacion.geojson")[0]).to_crs(CRS)
    declared_ha = pd.to_numeric(footprint["area_ha"]).sum()
    repaired = gpd.GeoSeries(footprint.geometry.apply(make_valid), crs=CRS)
    area_ha = repaired.area.sum() / M2_PER_HA
    if abs(area_ha - declared_ha) > GVA_AREA_TOLERANCE * declared_ha:
        raise ValueError(f"Huella GVA: superficie reparada {area_ha:.0f} ha distinta de la declarada {declared_ha:.0f} ha")
    return repaired


def build_flood_events() -> None:
    """Zonas inundadas de la DANA por gravedad y fuente.

    Copernicus EMS (satélite): todo lo que marca como agua o rastro de agua (observedEvent). Gravedad por calado
    (floodDepth): < 0,5 m Leve, 0,5-1 m Moderada, > 1 m Grave; lo que no tiene calado medido cuenta como Leve.
    GVA (huella oficial): lo que añade a Copernicus, sobre todo cascos urbanos donde el satélite no ve el agua entre
    edificios. No trae calado: cuenta como Leve y la gravedad de la celda se completa con los calados de campo.
    """
    def read(pattern: str) -> gpd.GeoDataFrame:
        """Une los shapefiles de todas las AOI de Copernicus EMS que cumplen el patrón."""
        layer = pd.concat([gpd.read_file(f).to_crs(CRS) for f in _files("copernicus_ems_EMSR773", pattern)])
        # Algunos productos (p. ej. AOI04 Algemesí) traen polígonos inválidos
        return layer.set_geometry(layer.geometry.make_valid())

    observed, depth = read("*observedEventA*.shp"), read("*floodDepthA*.shp")
    depth["severity"] = depth["value"].map(DEPTH_TO_SEVERITY)

    area = box(*_area())
    layers = {name: depth.loc[depth["severity"] == name].union_all() for name in ("Grave", "Moderada")}
    layers["Moderada"] = layers["Moderada"].difference(layers["Grave"])
    ems = observed.union_all()
    layers["Leve"] = ems.difference(layers["Grave"]).difference(layers["Moderada"])
    floods = gpd.GeoDataFrame(
        {"episode": EPISODE_NAME, "severity": list(layers), "source": "Copernicus EMS"},
        geometry=[geom.intersection(area) for geom in layers.values()],
        crs=CRS,
    )
    gva_only = gva_footprint().union_all().intersection(area).difference(ems)
    gva = gpd.GeoDataFrame({"episode": [EPISODE_NAME], "severity": ["Leve"], "source": ["GVA"]}, geometry=[gva_only], crs=CRS)
    pd.concat([floods, gva], ignore_index=True).to_file(_output("flood_events"), driver="GPKG")


def build_municipalities() -> None:
    """Recintos municipales del CNIG que tocan la zona de datos (para no asignar barrios de Valencia a otros municipios)."""
    area = gpd.GeoSeries([box(*_area())], crs=CRS)
    munis = gpd.read_file(_files("cnig_limites_municipales", "recintos_municipales_*.shp")[0], bbox=area).to_crs(CRS)
    munis = munis.rename(columns={"NATCODE": "codigo", "NAMEUNIT": "municipio"})[["codigo", "municipio", "geometry"]]
    munis.clip(area.iloc[0]).to_file(_output("municipalities"), driver="GPKG")


def build_channels() -> None:
    """Cauces principales: el nuevo cauce del Turia y los cursos naturales con nombre del IGN.

    Se descartan acequias y canales de riego (forman una red muy densa que no distingue nada), los tramos
    subterráneos (no pueden desbordarse en superficie; el IGN guarda así 10,5 km del antiguo cauce del Turia bajo
    el Jardí), los cursos sin nombre (desagües menores, como el de Natzaret) y el antiguo cauce con nombre, que no
    lleva agua desde que el río se desvió tras 1957.
    """
    courses = gpd.read_file(_files("ign_hidrografia", "watercourse.geojson")[0]).to_crs(CRS)
    name = courses["nombre"].fillna("")
    main = (
        (name.isin(MAIN_CHANNEL_NAMES) | ((courses["tipo_cursod"] == "Curso natural") & (name != "")))
        & (courses["situaciond"] == "En superficie")
        & ~name.isin(EXCLUDED_CHANNEL_NAMES)
    )
    channels = courses.loc[main, ["nombre", "tipo_cursod", "geometry"]].rename(columns={"tipo_cursod": "tipo"})
    channels.clip(box(*_area(margin_m=CHANNEL_MARGIN_M))).to_file(_output("channels"), driver="GPKG")


def hand_from_flowdir(flowdir: np.ndarray, dem: np.ndarray, drain: np.ndarray) -> np.ndarray:
    """Altura de cada píxel sobre el píxel en que acaba su agua siguiendo la dirección del flujo (D8).

    El camino acaba en un píxel de desagüe (drain: cauce o mar), en un punto sin salida (la Albufera, que no está a
    0 m) o en el borde del MDT por donde sale el agua. Se resuelve saltando de destino en destino: cada paso duplica la
    distancia recorrida, así que bastan unas decenas de pasos para todo el MDT.
    """
    rows, cols = flowdir.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    target = index.copy()
    for code, (dr, dc) in D8_MOVES.items():
        r, c = np.nonzero(flowdir == code)
        r2, c2 = r + dr, c + dc
        inside = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
        target[r[inside], c[inside]] = r2[inside] * cols + c2[inside]
    target = target.ravel()
    target[drain.ravel()] = index.ravel()[drain.ravel()]
    while not np.array_equal(jumped := target[target], target):
        target = jumped
    altitude = dem.ravel()
    return (altitude - altitude[target]).reshape(rows, cols)


def build_hand() -> None:
    """Altura sobre el cauce (HAND, Height Above Nearest Drainage) a partir del MDT y los cauces ya preparados.

    1. MDT remuestreado a HAND_RES_M (media) con los cauces principales y el mar rebajados HAND_BURN_M.
    2. pysheds rellena las depresiones, resuelve las zonas planas y calcula la dirección del flujo (D8).
    3. Cada píxel se sigue cuesta abajo hasta su desagüe y se resta la altitud (sin rebajar) de ese desagüe.
    Se guarda en centímetros enteros, como el MDT. Puede ser algo negativa en depresiones cerradas (por debajo del
    punto por donde rebosan).
    """
    with rasterio.open(_output("terrain")) as src:
        shape = (round(src.height * src.res[1] / HAND_RES_M), round(src.width * src.res[0] / HAND_RES_M))
        raw = src.read(1, out_shape=shape, resampling=Resampling.average, masked=True)
        dem = (raw.astype(float) * src.scales[0] + src.offsets[0]).filled(np.nan)
        transform = src.transform * src.transform.scale(src.width / shape[1], src.height / shape[0])
        profile, nodata = src.profile, src.nodata
    channels = gpd.read_file(_output("channels")).to_crs(profile["crs"])
    streams = rasterize(((g, 1) for g in channels.geometry), out_shape=shape, transform=transform, all_touched=True,
                        dtype="uint8").astype(bool)
    drain = (streams | (dem == SEA_ALTITUDE_M)) & ~np.isnan(dem)
    burned = np.where(drain, dem - HAND_BURN_M, dem)

    no_value = -9999.0
    with tempfile.TemporaryDirectory() as folder:
        burned_path = Path(folder) / "mdt_rebajado.tif"
        with rasterio.open(burned_path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                           dtype="float32", crs=profile["crs"], transform=transform, nodata=no_value) as dst:
            dst.write(np.where(np.isnan(burned), no_value, burned).astype("float32"), 1)
        grid = Grid.from_raster(str(burned_path))
        conditioned = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(grid.read_raster(str(burned_path)))))
        flowdir = np.asarray(grid.flowdir(conditioned))

    hand = hand_from_flowdir(flowdir, dem, drain)
    hand_cm = np.where(np.isnan(hand), nodata, np.round(hand / MDT_SCALE_M))
    profile = profile | {"height": shape[0], "width": shape[1], "transform": transform, "dtype": "int32"}
    with rasterio.open(_output("hand"), "w", **profile) as dst:
        dst.write(hand_cm.astype("int32"), 1)
        dst.scales = (MDT_SCALE_M,)


def build_field_depths() -> None:
    """Puntos con el calado medido en campo tras la DANA (GVA: IVE, SGRT y UPV). No se usan para entrenar: sirven
    para dar gravedad a las celdas que solo marca la huella de la GVA y para comprobar el modelo."""
    points = pd.concat([gpd.read_file(f).to_crs(CRS) for f in _files("gva_dana2024", "altura_agua_*.geojson")])
    points["depth_m"] = pd.to_numeric(points["altura_agua_m"].astype(str).str.replace(",", "."), errors="coerce")
    points = points.dropna(subset=["depth_m"]).rename(columns={"fuente_datos": "source"})
    points = points[points.within(box(*_area()))][["source", "depth_m", "geometry"]]
    gpd.GeoDataFrame(points, crs=CRS).to_file(_output("field_depths"), driver="GPKG")


def build_non_urban() -> None:
    """Suelo no urbano de Corine Land Cover 2018 (arrozales y agua permanente) en la zona de datos: esas celdas se excluyen."""
    cover = gpd.read_file(_files("copernicus_clc", "clc2018_no_urbano.geojson")[0]).to_crs(CRS)
    cover = cover[cover["Code_18"].astype(str).isin(NON_URBAN_CLC_CODES)]
    cover.clip(box(*_area()))[["Code_18", "geometry"]].to_file(_output("non_urban"), driver="GPKG")


def build_geology() -> None:
    """Geología del IGME (GEODE 50) agrupada en las unidades que importan para la inundación (config.GEOLOGY_GROUPS);
    el resto de unidades no se guarda."""
    geology = gpd.read_file(_files("igme_geode50", "geode50.geojson")[0]).to_crs(CRS)
    desc = geology["DESC_UNIT"].str.lower()
    geology["grupo"] = None
    for group, pattern in GEOLOGY_GROUPS.items():
        geology.loc[geology["grupo"].isna() & desc.str.contains(pattern, na=False), "grupo"] = group
    geology = geology.dropna(subset=["grupo"]).clip(box(*_area()))
    geology[["grupo", "DESC_UNIT", "geometry"]].to_file(_output("geology"), driver="GPKG")


def build_snczi() -> None:
    """Zona inundable del SNCZI (T=500 años), solo como referencia. El fichero es de toda España: se lee solo
    la zona. Viene en UTM 30N con coordenada Z, que se quita."""
    archive = _files("snczi", f"laminasPB-q{SNCZI_RETURN_PERIOD}.zip")[0]
    with zipfile.ZipFile(archive) as content:
        shapefile = next(name for name in content.namelist() if name.lower().endswith(".shp"))
    flood_zones = gpd.read_file(f"zip://{archive.as_posix()}!{shapefile}", bbox=_area())
    flood_zones = flood_zones.set_crs(CRS, allow_override=True)
    flood_zones = flood_zones.set_geometry(shapely.force_2d(flood_zones.geometry.values), crs=CRS)
    flood_zones = flood_zones.set_geometry(flood_zones.geometry.make_valid()).clip(box(*_area()))
    flood_zones[["RIO", "geometry"]].to_file(_output("snczi"), driver="GPKG")


def main() -> None:
    """Prepara todas las fuentes, en orden: de data/raw a data/preparados."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for step in (build_mdt, build_imperviousness, build_patricova, build_neighbourhoods, build_flood_events,
                 build_municipalities, build_channels, build_hand, build_field_depths, build_non_urban, build_geology,
                 build_snczi):
        step()
        log.info("%s hecho", step.__name__)


if __name__ == "__main__":
    main()
