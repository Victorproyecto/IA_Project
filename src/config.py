"""Configuración común del proyecto: rutas, sistema de referencia y parámetros del piloto."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
# raw (originales) --ingest--> preparados --run--> procesados --train--> modelos. Nada de data/ se versiona:
# raw se comparte por la nube y el resto se genera (docs/DATOS.md)
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "preparados"
PROCESSED_DIR = DATA_DIR / "procesados"
MODELS_DIR = DATA_DIR / "modelos"

# Ficheros de data/preparados: los genera la ingesta y los lee el pipeline (fuente -> fichero)
PREPARED_FILES = {
    "terrain": "mdt_altitud.tif",
    "imperviousness": "impermeabilidad_suelo.tif",
    "patricova": "patricova_peligrosidad.gpkg",
    "neighbourhoods": "barrios_valencia.gpkg",
    "flood_events": "zonas_inundadas_dana2024.gpkg",
    "municipalities": "municipios.gpkg",
    "channels": "cauces_principales.gpkg",
    "field_depths": "calados_campo_dana2024.gpkg",
    "non_urban": "suelo_no_urbano_clc2018.gpkg",
    "geology": "geologia_igme.gpkg",
    "snczi": "snczi_zona_inundable_t500.gpkg",
}

# Ficheros de data/procesados (los nombres son las tablas del modelo de datos, docs/modelo_datos.png)
ZONES_FILE = PROCESSED_DIR / "zones.parquet"
FLOOD_EVENTS_FILE = PROCESSED_DIR / "flood_events.parquet"
TRAINING_FILE = PROCESSED_DIR / "training_table.parquet"
# Celdas de la Ribera para la prueba externa: nunca entran en el entrenamiento ni en el mapa
EXTERNAL_TEST_FILE = PROCESSED_DIR / "external_test_ribera.parquet"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
DATA_SOURCES_FILE = PROCESSED_DIR / "data_sources.csv"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"
EXPERIMENTS_FILE = MODELS_DIR / "experiments.jsonl"

# Todo el trabajo espacial se hace en ETRS89 / UTM 30N; el mapa usa WGS84
CRS = "EPSG:25830"
CRS_WEB = "EPSG:4326"

# Municipios del área piloto: València y las comarcas de L'Horta Sud y L'Horta Oest (delimitación habitual; las
# comarcas no son una división administrativa oficial), es decir, la cuenca del Poyo y lo que inundó. La ciudad apenas
# se inundó: el agua llegó por el Poyo. Se cubren los términos municipales enteros.
PILOT_MUNICIPALITIES = [
    "València",
    # L'Horta Sud
    "Albal", "Alcàsser", "Alfafar", "Benetússer", "Beniparrell", "Catarroja", "Llocnou de la Corona", "Massanassa",
    "Paiporta", "Picassent", "Sedaví", "Silla",
    # L'Horta Oest
    "Alaquàs", "Aldaia", "Manises", "Mislata", "Paterna", "Picanya", "Quart de Poblet", "Torrent", "Xirivella",
]
# Rectángulo en ETRS89 / UTM 30N (x_min, y_min, x_max, y_max) que contiene esos municipios enteros; dentro solo se quedan
# sus celdas. Coordenadas múltiplos de 100 y 250 m para que ningún tamaño de celda recorte el borde. Las áreas solo se
# definen aquí, en metros: lo que necesita grados (geocodificación, recortes) se calcula a partir de ellas.
PILOT_EXTENT = (705000, 4351000, 735500, 4383500)
# Prueba externa: la Ribera (Magro y Xúquer: Algemesí, l'Alcúdia, Carlet, Guadassuar...), otra cuenca. El modelo nunca
# la ve: solo se predice para medir si generaliza. Llega hasta el límite del MDT descargado (39,16° N)
EXTERNAL_TEST_EXTENT = (714000, 4338500, 741000, 4352000)
# 100 m: mismo rendimiento que 250 m con mucho más detalle; a 50 m el modelo empeora (docs/MEJORAS_MODELO_NF3.md)
CELL_SIZE_M = 100
# El MDT05 guarda el mar como altitud 0 exacta, no como nodata
SEA_ALTITUDE_M = 0
# Celdas con al menos esta fracción de mar (según el MDT) se descartan
MAX_SEA_FRACTION = 0.5
# Celdas con al menos esta fracción de suelo no urbano según Corine Land Cover 2018 se descartan: arrozales (213) y agua
# permanente (511 cauces, 512 embalses, 521 lagunas costeras como la Albufera, 523 mar). No es suelo urbano, el satélite
# no distingue esa agua de la inundación y son aciertos fáciles que inflan las métricas
NON_URBAN_CLC_CODES = ["213", "511", "512", "521", "522", "523"]
MAX_NON_URBAN_FRACTION = 0.5

# Episodio de referencia. La etiqueta une la zona inundada de Copernicus EMS (satélite) y la huella oficial de la GVA:
# el satélite no ve el agua entre edificios y la GVA sí cubre los cascos urbanos de L'Horta Sud
EPISODE_NAME = "DANA 29-10-2024"
# Fracción mínima de la celda cubierta por agua para contarla como inundada (evita bordes rozados)
FLOOD_MIN_FRACTION = 0.10
SEVERITY_ORDER = ["Leve", "Moderada", "Grave"]

# Validación cruzada por bloques espaciales de ~1 km repartidos en N_FOLDS grupos
BLOCK_SIZE_CELLS = round(1000 / CELL_SIZE_M)
N_FOLDS = 5

# Municipio cuyas celdas reciben barrio y distrito (las de otros municipios solo llevan el municipio)
CITY_MUNICIPALITY = "València"
# Buscador de la aplicación: CartoCiudad (IGN) solo sugiere calles y direcciones con estos códigos postales, los del
# área piloto, obtenidos con la geocodificación inversa de CartoCiudad sobre una malla de puntos cada 500 m.
SEARCH_POSTAL_CODES = [
    "46001", "46002", "46003", "46004", "46005", "46006", "46007", "46008", "46009", "46010", "46011", "46012", "46013",
    "46014", "46015", "46016", "46017", "46018", "46019", "46020", "46021", "46022", "46023", "46024", "46025", "46026",
    "46035", "46111", "46112", "46131", "46135", "46182", "46184", "46200", "46210", "46220", "46225", "46229", "46290",
    "46460", "46469", "46470", "46900", "46901", "46909", "46910", "46920", "46930", "46940", "46950", "46960", "46970",
    "46980", "46988", "46989",
]
# Cauces principales para la distancia al cauce: el nuevo cauce del Turia y los cursos naturales con nombre del IGN.
# Se excluyen los cursos sin nombre (desagües menores, como el de Natzaret) y el antiguo cauce del Turia, que ya no
# lleva agua (el río se desvió tras la riada de 1957).
MAIN_CHANNEL_NAMES = ["Canal del Riu Túria"]
EXCLUDED_CHANNEL_NAMES = ["Antiguo Cauce del Rio Túria"]

RISK_LEVELS = ["Bajo", "Medio", "Alto"]
# Nivel Alto: las celdas de mayor probabilidad que concentran esta fracción de lo inundado (Medio + Alto concentran
# TARGET_RECALL), sin depender de cuánto se inundó el área.
ALTO_RECALL = 0.50

# Geología (IGME, GEODE 50): fracción de la celda en cada grupo de unidades, según palabras de su descripción
GEOLOGY_GROUPS = {
    "geo_aluvial": "llanura de inundaci|fondo de valle|fondos de valle|playa aluvial|aluvial coluvial",
    "geo_marjal": "albufera|turbera|marjal",
    "geo_abanico": "abanico|cono de deyecci",
    "geo_glacis": "glacis|coluvi",
}
GEOLOGY_LABELS = {"geo_aluvial": "llanura de inundación o fondo de valle", "geo_marjal": "limos de albufera o marjal",
                  "geo_abanico": "abanico aluvial", "geo_glacis": "glacis o coluvión"}
# En la ficha de la celda, su geología es el grupo que más ocupa si llega a esta fracción; si no, "otros"
GEOLOGY_DOMINANT_MIN_FRACTION = 0.5
# Variables del modelo: terreno, suelo, cauces y geología. Los mapas oficiales de peligrosidad (PATRICOVA y SNCZI) no son
# variables: son la referencia con la que se compara
FEATURES = [
    "altitude_avg",
    "slope_avg",
    "imperviousness",
    "dist_channel_km",
    *GEOLOGY_GROUPS,
]
# Calado medido en campo a partir del cual un punto cuenta como "con agua" (comprobación con calados de campo)
FIELD_DEPTH_MIN_M = 0.3
# Calado (m) que separa Leve / Moderada / Grave, igual que en Copernicus EMS
SEVERITY_DEPTH_LIMITS = (0.5, 1.0)
# Separación mínima (m) entre el área de entrenamiento y la de prueba en la prueba de generalización entre zonas
GENERALIZATION_GAP_M = 1000
# SNCZI (MITECO): zona inundable para el periodo de retorno de 500 años, comparable a un episodio extremo como la DANA.
# Solo como referencia
SNCZI_RETURN_PERIOD = 500
SNCZI_COLUMN = f"snczi_t{SNCZI_RETURN_PERIOD}"
# PATRICOVA: niveles de peligrosidad 1 (máxima) a 6 (mínima); 0 = fuera de toda zona de peligrosidad
PATRICOVA_MAX_LEVEL = 6

# Criterios de éxito (Fase 1, 4.4)
TARGET_RECALL = 0.80
TARGET_F1 = 0.70

RANDOM_STATE = 42
