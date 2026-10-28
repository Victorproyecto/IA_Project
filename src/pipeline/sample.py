"""Datos de muestra simulados para que el MVP funcione de punta a punta sin las fuentes reales.

Todo lo que sale de aquí se marca como simulado (DataSource.status y episodes.is_simulated).
Los valores imitan la forma de Valencia (llana, baja hacia el mar, centro muy impermeable),
pero no son observaciones reales ni inundaciones históricas.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.config import CRS, CRS_WEB, RANDOM_STATE

# Centroides aproximados de los distritos de Valencia dentro del área piloto (lat, lon)
DISTRICTS = {
    "Ciutat Vella": (39.4747, -0.3765),
    "L'Eixample": (39.4655, -0.3720),
    "Extramurs": (39.4700, -0.3880),
    "Campanar": (39.4850, -0.3980),
    "La Saïdia": (39.4880, -0.3750),
    "El Pla del Real": (39.4790, -0.3600),
    "L'Olivereta": (39.4690, -0.4030),
    "Patraix": (39.4600, -0.3970),
    "Jesús": (39.4500, -0.3880),
    "Quatre Carreres": (39.4500, -0.3620),
    "Poblats Marítims": (39.4630, -0.3300),
    "Camins al Grau": (39.4650, -0.3470),
    "Algirós": (39.4770, -0.3460),
    "Benimaclet": (39.4870, -0.3600),
    "Rascanya": (39.4970, -0.3700),
    "Benicalap": (39.4950, -0.3930),
    "Pobles de l'Oest": (39.4900, -0.4250),
}

# Estaciones simuladas: id, nombre, lat, lon
STATIONS = [
    ("SIM-01", "Centro (simulada)", 39.4806, -0.3664),
    ("SIM-02", "Oeste (simulada)", 39.4750, -0.4200),
    ("SIM-03", "Sur (simulada)", 39.4400, -0.3700),
    ("SIM-04", "Litoral (simulada)", 39.4600, -0.3300),
]

# Episodios de muestra: nombre, inicio, lluvia máxima en 24 h (mm), sesgo espacial (u, v) de la tormenta
EPISODES = [
    ("Episodio simulado 2016-12", "2016-12-03 18:00", 95, (0.7, 0.3)),
    ("Episodio simulado 2017-01", "2017-01-19 06:00", 60, (0.5, 0.5)),
    ("Episodio simulado 2017-03", "2017-03-26 12:00", 130, (0.4, 0.2)),
    ("Episodio simulado 2018-10", "2018-10-18 00:00", 75, (0.8, 0.6)),
    ("Episodio simulado 2019-04", "2019-04-20 15:00", 45, (0.3, 0.7)),
    ("Episodio simulado 2019-09", "2019-09-12 03:00", 160, (0.6, 0.1)),
    ("Episodio simulado 2020-01", "2020-01-19 21:00", 110, (0.9, 0.5)),
    ("Episodio simulado 2021-05", "2021-05-08 10:00", 35, (0.5, 0.5)),
    ("Episodio simulado 2022-03", "2022-03-21 08:00", 120, (0.5, 0.3)),
    ("Episodio simulado 2022-11", "2022-11-14 14:00", 70, (0.2, 0.4)),
    ("Episodio simulado 2023-09", "2023-09-03 05:00", 140, (0.7, 0.2)),
    ("Episodio simulado 2023-11", "2023-11-27 17:00", 55, (0.4, 0.6)),
    ("DANA 29-10-2024 (muestra simulada)", "2024-10-29 06:00", 190, (0.4, 0.0)),
]

# Escenarios predefinidos: solo lluvia, sin inundaciones observadas
SCENARIOS = [
    ("Escenario: lluvia moderada (40 mm/24 h)", 40, (0.5, 0.5)),
    ("Escenario: lluvia torrencial (150 mm/24 h)", 150, (0.5, 0.3)),
]

EPISODE_HOURS = 48


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(RANDOM_STATE + offset)


def _unit_coords(zones: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas normalizadas 0-1 de cada celda: u crece hacia el este, v hacia el norte."""
    c = zones.geometry.centroid
    u = (c.x - c.x.min()) / (c.x.max() - c.x.min())
    v = (c.y - c.y.min()) / (c.y.max() - c.y.min())
    return u.to_numpy(), v.to_numpy()


def _smooth_noise(u: np.ndarray, v: np.ndarray, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    phases = rng.uniform(0, 2 * np.pi, size=4)
    return amplitude * (
        np.sin(7 * u + phases[0]) * np.cos(5 * v + phases[1]) + 0.5 * np.sin(13 * u + 11 * v + phases[2])
    )


def _turia_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Distancia normalizada al antiguo cauce del Turia, que cruza la ciudad de oeste a este."""
    return np.abs(v - (0.62 - 0.18 * u + 0.05 * np.sin(6 * u)))


def terrain(zones: gpd.GeoDataFrame) -> np.ndarray:
    """altitude_avg (m): baja de ~20 m al oeste a ~1 m en la costa, con el cauce y una depresión al sur."""
    u, v = _unit_coords(zones)
    rng = _rng(1)
    altitude = 1 + 19 * (1 - u) ** 1.3
    altitude -= 3 * np.exp(-((_turia_distance(u, v) / 0.03) ** 2))
    altitude -= 2.5 * np.exp(-(((u - 0.6) / 0.2) ** 2 + ((v - 0.1) / 0.15) ** 2))
    altitude += _smooth_noise(u, v, rng, 0.8) + rng.normal(0, 0.3, len(u))
    return np.clip(altitude, 0, None).round(2)


def imperviousness(zones: gpd.GeoDataFrame) -> np.ndarray:
    """imperviousness (%): máximo en el centro, jardín del Turia y huerta periférica más permeables."""
    u, v = _unit_coords(zones)
    rng = _rng(2)
    value = 35 + 60 * np.exp(-(((u - 0.5) / 0.35) ** 2 + ((v - 0.5) / 0.35) ** 2))
    value -= 50 * np.exp(-((_turia_distance(u, v) / 0.025) ** 2))
    value += rng.normal(0, 6, len(u))
    return np.clip(value, 0, 100).round(1)


def patricova(zones: gpd.GeoDataFrame, altitude: np.ndarray) -> np.ndarray:
    """patricova_level: 0 sin peligrosidad, 1 (máxima) a 6 (mínima), concentrado en zonas bajas."""
    u, v = _unit_coords(zones)
    rng = _rng(3)
    hazard = (8 - altitude) / 8 + 0.6 * np.exp(-(((u - 0.6) / 0.2) ** 2 + ((v - 0.1) / 0.15) ** 2))
    hazard += rng.normal(0, 0.15, len(u))
    level = np.where(hazard > 0.35, np.clip(np.round(7 - 6 * hazard), 1, 6), 0)
    return level.astype(int)


def districts(zones: gpd.GeoDataFrame) -> np.ndarray:
    """Distrito aproximado por centroide más cercano (en modo real se usa la capa del Ayuntamiento)."""
    names = list(DISTRICTS)
    points = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in DISTRICTS.values()], crs=CRS_WEB
    ).to_crs(CRS)
    centroids = zones.geometry.centroid
    distances = np.stack([centroids.distance(p).to_numpy() for p in points], axis=1)
    return np.array(names)[distances.argmin(axis=1)]


def stations() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [{"station_id": s, "name": n, "geometry": Point(lon, lat)} for s, n, lat, lon in STATIONS],
        crs=CRS_WEB,
    ).to_crs(CRS)


def episodes() -> pd.DataFrame:
    """Catálogo de episodios históricos (simulados) y escenarios predefinidos."""
    rows = [
        {"episode_id": i + 1, "name": name, "started_at": pd.Timestamp(start), "kind": "historical"}
        for i, (name, start, _, _) in enumerate(EPISODES)
    ]
    base = pd.Timestamp("2030-01-01 00:00")
    rows += [
        {"episode_id": len(EPISODES) + i + 1, "name": name, "started_at": base, "kind": "scenario"}
        for i, (name, _, _) in enumerate(SCENARIOS)
    ]
    df = pd.DataFrame(rows)
    df["ended_at"] = df["started_at"] + pd.Timedelta(hours=EPISODE_HOURS - 1)
    df["is_simulated"] = True
    return df


def station_rain(station_gdf: gpd.GeoDataFrame, episode_df: pd.DataFrame) -> pd.DataFrame:
    """Lluvia horaria por estación y episodio: station_id, episode_id, observed_at, precipitation_mm, temperature_c."""
    rng = _rng(4)
    storms = {i + 1: (peak, bias) for i, (_, _, peak, bias) in enumerate(EPISODES)}
    storms.update({len(EPISODES) + i + 1: (peak, bias) for i, (_, peak, bias) in enumerate(SCENARIOS)})

    xy = np.c_[station_gdf.geometry.x, station_gdf.geometry.y]
    su = (xy[:, 0] - xy[:, 0].min()) / np.ptp(xy[:, 0])
    sv = (xy[:, 1] - xy[:, 1].min()) / np.ptp(xy[:, 1])

    rows = []
    for _, ep in episode_df.iterrows():
        peak, (bu, bv) = storms[ep["episode_id"]]
        # El núcleo de la tormenta cae donde indica el sesgo; las estaciones lejanas reciben menos
        spatial = np.exp(-((su - bu) ** 2 + (sv - bv) ** 2) / 0.5)
        center, width = rng.uniform(14, 30), rng.uniform(3, 7)
        hours = np.arange(EPISODE_HOURS)
        profile = np.exp(-0.5 * ((hours - center) / width) ** 2)
        profile /= profile.sum()
        for k, station_id in enumerate(station_gdf["station_id"]):
            rain = peak * spatial[k] * profile * rng.lognormal(0, 0.15, EPISODE_HOURS)
            for h in hours:
                rows.append(
                    {
                        "station_id": station_id,
                        "episode_id": ep["episode_id"],
                        "observed_at": ep["started_at"] + pd.Timedelta(hours=int(h)),
                        "precipitation_mm": round(float(rain[h]), 2),
                        "temperature_c": round(float(rng.normal(17, 2)), 1),
                    }
                )
    return pd.DataFrame(rows)


def flood_labels(table: pd.DataFrame) -> pd.Series:
    """Etiqueta simulada de inundación por celda y episodio a partir de las variables físicas.

    Sustituye a Copernicus EMS mientras no haya cartografía real; devuelve la gravedad
    ("Leve", "Moderada", "Grave") o vacío si la celda no se inundó.
    """
    rng = _rng(5)
    latent = (
        0.045 * table["precipitation_24h"]
        + 0.06 * table["precipitation_1h"]
        - 0.30 * table["altitude_avg"]
        + 0.025 * table["imperviousness"]
        + 0.45 * table["patricova_score"]
        - 0.60 * table["slope_avg"]
        + rng.normal(0, 1.0, len(table))
        - 7.0
    )
    severity = pd.cut(latent, bins=[0, 1.5, 3, np.inf], labels=["Leve", "Moderada", "Grave"])
    return severity.astype("string")
