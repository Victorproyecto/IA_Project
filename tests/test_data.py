"""Tests de datos sobre los Parquet generados por el pipeline con los datos reales."""

import geopandas as gpd
import pandas as pd

from src import config
from src.pipeline import checks


def test_processed_files_pass_data_checks(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    events = pd.read_parquet(config.FLOOD_EVENTS_FILE)
    training = pd.read_parquet(config.TRAINING_FILE)

    assert checks.check_zones(zones) == []
    assert checks.check_flood_events(events, zones) == []
    assert checks.check_training(training, config.FEATURES) == []


def test_checks_detect_bad_data(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    events = pd.read_parquet(config.FLOOD_EVENTS_FILE).head(5).copy()
    events.loc[0, "zone_id"] = 999_999
    events.loc[1, "flooded_fraction"] = 1.5
    errors = checks.check_flood_events(events, zones)
    assert any("zone_id" in e for e in errors)
    assert any("flooded_fraction" in e for e in errors)


def test_gva_footprint_is_repaired_to_the_declared_area():
    """La huella de la GVA llega con geometría inválida; tras repararla debe medir lo que declara la GVA (42.550 ha).

    Necesita la descarga original (data/raw), que no se versiona: sin ella, el test se salta."""
    import pytest

    from src.pipeline import ingest

    declared_ha = 42_550
    try:
        footprint = ingest.gva_footprint()
    except FileNotFoundError:
        pytest.skip("falta la huella de la GVA en data/raw/gva_dana2024")
    assert footprint.is_valid.all()
    area_ha = footprint.area.sum() / ingest.M2_PER_HA
    assert abs(area_ha - declared_ha) <= ingest.GVA_AREA_TOLERANCE * declared_ha


def test_label_combines_copernicus_and_gva(built_data):
    events = pd.read_parquet(config.FLOOD_EVENTS_FILE)
    sources = set(" y ".join(events["source"]).split(" y "))
    assert sources == {"Copernicus EMS", "GVA"}


def test_external_test_is_separate_from_the_pilot(built_data):
    """La Ribera (prueba externa) no comparte celdas con el área piloto ni entra en el entrenamiento."""
    zones = gpd.read_parquet(config.ZONES_FILE)
    external = gpd.read_parquet(config.EXTERNAL_TEST_FILE)
    training = pd.read_parquet(config.TRAINING_FILE)
    assert not set(external["zone_id"]) & set(zones["zone_id"])
    assert not set(external["zone_id"]) & set(training["zone_id"])
    # Ninguna celda de la prueba externa cae sobre una celda del área piloto (se separan por municipio)
    centroids = external[["zone_id"]].set_geometry(external.geometry.centroid)
    assert gpd.sjoin(centroids, zones[["geometry"]], predicate="within").empty
    assert set(config.FEATURES) <= set(external.columns) and external["flooded"].isin([0, 1]).all()


def test_pilot_and_external_areas_follow_the_municipality_list(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    external = gpd.read_parquet(config.EXTERNAL_TEST_FILE)
    assert zones["municipality"].isin(config.PILOT_MUNICIPALITIES).all()
    assert not external["municipality"].isin(config.PILOT_MUNICIPALITIES).any()


def test_geology_fractions_are_valid(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    fractions = zones[list(config.GEOLOGY_GROUPS)]
    assert fractions.stack().between(0, 1).all()
    assert (fractions.sum(axis=1) <= 1.01).all()  # los grupos no se solapan


def test_flooded_cells_match_label_threshold(built_data):
    events = pd.read_parquet(config.FLOOD_EVENTS_FILE)
    training = pd.read_parquet(config.TRAINING_FILE)
    assert (events["flooded_fraction"] >= config.FLOOD_MIN_FRACTION).all()
    assert set(events["zone_id"]) == set(training.loc[training["flooded"] == 1, "zone_id"])


def test_predictions_reference_existing_zones_and_are_probabilities(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    predictions = pd.read_parquet(config.PREDICTIONS_DIR)
    assert predictions["zone_id"].isin(zones["zone_id"]).all()
    assert predictions["zone_id"].is_unique
    assert predictions[["risk_probability", "oof_probability"]].stack().between(0, 1).all()
    assert set(predictions["risk_level"]) <= set(config.RISK_LEVELS)


def test_data_sources_document_every_source(built_data):
    sources = pd.read_csv(config.DATA_SOURCES_FILE)
    assert {"name", "url", "license", "file", "updated_at"} <= set(sources.columns)
