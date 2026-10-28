"""Configuración común del proyecto: rutas, sistema de referencia y parámetros del piloto."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ficheros de data/processed (modelo de datos de la Fase 1, sección 7)
ZONES_FILE = PROCESSED_DIR / "zones.parquet"
WEATHER_DIR = PROCESSED_DIR / "weather_observations"
FLOOD_EVENTS_FILE = PROCESSED_DIR / "flood_events.parquet"
EPISODES_FILE = PROCESSED_DIR / "episodes.parquet"
TRAINING_FILE = PROCESSED_DIR / "training_table.parquet"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
DATA_SOURCES_FILE = PROCESSED_DIR / "data_sources.csv"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

# Todo el trabajo espacial se hace en ETRS89 / UTM 30N; el mapa usa WGS84
CRS = "EPSG:25830"
CRS_WEB = "EPSG:4326"

# Ciudad de Valencia (núcleo urbano) en WGS84: lon_min, lat_min, lon_max, lat_max
CITY_BBOX = (-0.43, 39.43, -0.32, 39.51)
CELL_SIZE_M = 250

# Distancia máxima de asignación celda-estación (Fase 1, 9.4 sesgo espacial de la lluvia)
MAX_STATION_DIST_M = 15_000

RISK_LEVELS = ["Bajo", "Medio", "Alto"]
FEATURES = [
    "altitude_avg",
    "slope_avg",
    "imperviousness",
    "patricova_score",
    "station_dist_m",
    "precipitation_1h",
    "precipitation_24h",
]

# Criterios de éxito (Fase 1, 4.4)
TARGET_RECALL = 0.80
TARGET_F1 = 0.70

RANDOM_STATE = 42

AEMET_API_KEY = os.getenv("AEMET_API_KEY", "")
AEMET_VALENCIA_MUNICIPIO = "46250"
