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

# Ficheros de data/processed
ZONES_FILE = PROCESSED_DIR / "zones.parquet"
FLOOD_EVENTS_FILE = PROCESSED_DIR / "flood_events.parquet"
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
# Celdas con al menos esta fracción de mar (según el MDT) se descartan
MAX_SEA_FRACTION = 0.5

# Episodio de referencia: la etiqueta sale de su mapa de zonas inundadas (Copernicus EMS EMSR773)
EPISODE_NAME = "DANA 29-10-2024"
# Fracción mínima de la celda cubierta por agua para contarla como inundada (evita bordes rozados)
FLOOD_MIN_FRACTION = 0.10
SEVERITY_ORDER = ["Leve", "Moderada", "Grave"]

# Validación cruzada por bloques espaciales: bloques de 4 x 4 celdas (1 km) repartidos en N_FOLDS grupos
BLOCK_SIZE_CELLS = 4
N_FOLDS = 5

RISK_LEVELS = ["Bajo", "Medio", "Alto"]
FEATURES = [
    "altitude_avg",
    "slope_avg",
    "imperviousness",
    "patricova_score",
]

# Criterios de éxito (Fase 1, 4.4)
TARGET_RECALL = 0.80
TARGET_F1 = 0.70

RANDOM_STATE = 42
