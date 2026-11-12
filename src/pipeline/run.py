"""Pipeline completo: ingesta, limpieza, variables, etiquetas y bloques espaciales. Uso: python -m src.pipeline.run"""

import logging

import pandas as pd

from src.config import (
    DATA_SOURCES_FILE, EXTERNAL_TEST_EXTENT, EXTERNAL_TEST_FILE, FEATURES, FLOOD_EVENTS_FILE, FLOOD_MIN_FRACTION,
    GEOLOGY_DOMINANT_MIN_FRACTION, GEOLOGY_GROUPS, GEOLOGY_LABELS, MAX_NON_URBAN_FRACTION, MAX_SEA_FRACTION, PILOT_EXTENT,
    PILOT_MUNICIPALITIES, PROCESSED_DIR, SNCZI_COLUMN, TRAINING_FILE, ZONES_FILE,
)
from src.pipeline import blocks, checks, features, sources
from src.pipeline.grid import build_grid, slope_from_altitude

log = logging.getLogger("pipeline")

# Los identificadores de las celdas de la prueba externa empiezan aquí, para no confundirlos con los del área piloto
EXTERNAL_ID_OFFSET = 1_000_000


def build_zones(extent: tuple[float, float, float, float] = PILOT_EXTENT, pilot: bool = True):
    """Celdas de tierra con terreno, suelo, cauces, geología, municipio / barrio y las referencias oficiales.

    Se descartan las celdas de mar y las de suelo no urbano según Corine Land Cover (arrozales y agua permanente, como
    el lago de la Albufera): su etiqueta es dudosa y serían aciertos fáciles. En el área piloto solo se quedan los
    municipios de PILOT_MUNICIPALITIES; en la prueba externa, al revés, se quitan.
    """
    zones = build_grid(extent)
    altitude, sea_fraction = sources.load_altitude(zones)
    zones["altitude_avg"] = altitude.round(2)
    zones["slope_avg"] = slope_from_altitude(zones).round(3)
    zones["imperviousness"] = sources.load_imperviousness(zones).round(1)
    zones["dist_channel_km"], zones["channel_name"] = sources.load_channel_distance(zones)
    zones["dist_channel_km"] = zones["dist_channel_km"].round(3)
    zones = zones.join(sources.load_geology(zones))
    # Grupo geológico que más ocupa la celda (para la ficha)
    geology = zones[list(GEOLOGY_GROUPS)]
    dominant = geology.idxmax(axis=1).map(GEOLOGY_LABELS)
    zones["geology"] = dominant.where(geology.max(axis=1) >= GEOLOGY_DOMINANT_MIN_FRACTION, "otros")
    # Mapas oficiales de peligrosidad: solo referencia, no son variables del modelo
    zones["patricova_level"] = sources.load_patricova(zones)
    zones["patricova_score"] = features.patricova_score(zones["patricova_level"])
    zones[SNCZI_COLUMN] = sources.load_snczi(zones)
    zones["municipality"], zones["district"], zones["neighbourhood"] = sources.load_admin_units(zones)

    land = (sea_fraction < MAX_SEA_FRACTION) & zones[FEATURES].notna().all(axis=1)
    non_urban = sources.load_non_urban_fraction(zones) >= MAX_NON_URBAN_FRACTION
    in_pilot = zones["municipality"].isin(PILOT_MUNICIPALITIES).to_numpy()
    in_area = in_pilot if pilot else ~in_pilot
    keep = land & ~non_urban & in_area
    log.info("celdas descartadas por ser mar: %d; por ser arrozal o agua permanente: %d; por municipio: %d",
             (~land).sum(), (land & non_urban).sum(), (land & ~non_urban & ~in_area).sum())
    return zones[keep].reset_index(drop=True)


def build_labels(zones) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (training_table, flood_events): una fila por celda con su etiqueta de la DANA."""
    observed = sources.load_observed_floods(zones)
    observed["flooded"] = (observed["flooded_fraction"] >= FLOOD_MIN_FRACTION).astype(int)
    events = observed[observed["flooded"] == 1].drop(columns="flooded").reset_index(drop=True)

    table = zones[["zone_id", "row", "col", *FEATURES]].merge(observed[["zone_id", "flooded"]], on="zone_id")
    table["block_id"] = blocks.block_ids(table["row"], table["col"])
    table["fold"] = blocks.assign_folds(table["block_id"], table["flooded"])
    return table[["zone_id", "block_id", "fold", *FEATURES, "flooded"]], events


def build_external_test():
    """Celdas de la Ribera con sus variables y su etiqueta, para la prueba externa (nunca para entrenar)."""
    zones = build_zones(EXTERNAL_TEST_EXTENT, pilot=False)
    zones["zone_id"] += EXTERNAL_ID_OFFSET
    observed = sources.load_observed_floods(zones)
    zones["flooded"] = (observed["flooded_fraction"] >= FLOOD_MIN_FRACTION).astype(int).to_numpy()
    return zones


def main() -> None:
    """Prepara el área piloto y la prueba externa, comprueba los datos y escribe los Parquet de data/procesados."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Zonas del área piloto con sus variables, referencias y municipio / barrio
    zones = build_zones()
    log.info("zonas: %d celdas", len(zones))

    # Etiquetas de la DANA y reparto de los bloques en grupos de validación cruzada
    training, events = build_labels(zones)
    log.info("entrenamiento: %d celdas, %.1f %% inundadas", len(training), 100 * training["flooded"].mean())
    log.info("grupos: %s", training.groupby("fold")["flooded"].agg(["size", "sum"]).to_dict("index"))

    # Prueba externa: la Ribera, otra cuenca que el modelo no ve
    external = build_external_test()
    log.info("prueba externa (Ribera): %d celdas, %.1f %% inundadas", len(external), 100 * external["flooded"].mean())

    # Comprobaciones de datos antes de escribir nada
    errors = (checks.check_zones(zones) + checks.check_flood_events(events, zones) + checks.check_training(training, FEATURES)
              + checks.check_zones(external))
    if errors:
        raise ValueError("Fallan las comprobaciones de datos:\n" + "\n".join(errors))

    # Escritura en Parquet / GeoParquet
    zones.to_parquet(ZONES_FILE)
    events.to_parquet(FLOOD_EVENTS_FILE)
    training.to_parquet(TRAINING_FILE)
    external.to_parquet(EXTERNAL_TEST_FILE)
    sources.data_sources_table(pd.Timestamp.now().floor("s")).to_csv(DATA_SOURCES_FILE, index=False)


if __name__ == "__main__":
    main()
