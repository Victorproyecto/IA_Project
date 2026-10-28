"""Tests de datos sobre los Parquet generados por el pipeline con la muestra."""

import geopandas as gpd
import pandas as pd

from src import config
from src.pipeline import checks


def test_processed_files_pass_data_checks(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    episodes = pd.read_parquet(config.EPISODES_FILE)
    weather = pd.read_parquet(config.WEATHER_DIR)
    events = pd.read_parquet(config.FLOOD_EVENTS_FILE)
    training = pd.read_parquet(config.TRAINING_FILE)

    assert checks.check_zones(zones) == []
    assert checks.check_weather(weather, zones, episodes) == []
    assert checks.check_flood_events(events, zones, episodes) == []
    assert checks.check_training(training, config.FEATURES) == []


def test_checks_detect_bad_data(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    episodes = pd.read_parquet(config.EPISODES_FILE)
    weather = pd.read_parquet(config.WEATHER_DIR).head(10).copy()
    weather.loc[0, "precipitation_mm"] = -5
    weather.loc[1, "zone_id"] = 999_999
    errors = checks.check_weather(weather, zones, episodes)
    assert any("negativa" in e for e in errors)
    assert any("zone_id" in e for e in errors)


def test_predictions_reference_existing_zones_and_are_probabilities(built_data):
    zones = gpd.read_parquet(config.ZONES_FILE)
    predictions = pd.read_parquet(config.PREDICTIONS_DIR)
    assert predictions["zone_id"].isin(zones["zone_id"]).all()
    assert predictions["risk_probability"].between(0, 1).all()
    assert set(predictions["risk_level"]) <= set(config.RISK_LEVELS)


def test_scenarios_are_marked_simulated(built_data):
    episodes = pd.read_parquet(config.EPISODES_FILE)
    predictions = pd.read_parquet(config.PREDICTIONS_DIR)
    scenario_ids = episodes.loc[episodes["kind"] == "scenario", "episode_id"]
    assert predictions.loc[predictions["episode_id"].isin(scenario_ids), "is_simulated"].all()


def test_data_sources_document_every_source(built_data):
    sources = pd.read_csv(config.DATA_SOURCES_FILE)
    assert {"name", "url", "license", "status", "updated_at"} <= set(sources.columns)
    assert sources["status"].isin(["real", "simulado"]).all()
