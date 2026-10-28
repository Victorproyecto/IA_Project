"""Pipeline completo: ingesta, limpieza, variables y etiquetas. Uso: python -m src.pipeline.run"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    AEMET_API_KEY, DATA_SOURCES_FILE, EPISODES_FILE, FEATURES, FLOOD_EVENTS_FILE, PROCESSED_DIR,
    TRAINING_FILE, WEATHER_DIR, ZONES_FILE,
)
from src.pipeline import aemet, checks, features, sample, sources
from src.pipeline.grid import build_grid, slope_from_altitude

log = logging.getLogger("pipeline")


def build_zones():
    zones = build_grid()
    zones["altitude_avg"] = sources.load_altitude(zones).round(2)
    zones["slope_avg"] = slope_from_altitude(zones).round(3)
    zones["imperviousness"] = sources.load_imperviousness(zones).round(1)
    zones["patricova_level"] = sources.load_patricova(zones, zones["altitude_avg"].to_numpy())
    zones["patricova_score"] = features.patricova_score(zones["patricova_level"])
    zones["district"], zones["neighbourhood"] = sources.load_neighbourhoods(zones)

    stations = sources.load_stations()
    zones = zones.merge(features.nearest_station(zones, stations), on="zone_id")
    return zones, stations


def add_aemet_forecast(episodes: pd.DataFrame, station_rain: pd.DataFrame, stations) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Añade la previsión horaria de AEMET como episodio "forecast" igual para todas las estaciones."""
    forecast = aemet.hourly_forecast()
    episode_id = episodes["episode_id"].max() + 1
    started = forecast["observed_at"].min()
    episode = pd.DataFrame([{
        "episode_id": episode_id,
        "name": f"Previsión AEMET {started:%d-%m-%Y %H:%M}",
        "started_at": started,
        "ended_at": forecast["observed_at"].max(),
        "kind": "forecast",
        "is_simulated": False,
    }])
    rain = pd.concat([forecast.assign(station_id=s, episode_id=episode_id) for s in stations["station_id"]])
    return pd.concat([episodes, episode], ignore_index=True), pd.concat([station_rain, rain], ignore_index=True)


def build_labels(zones, peaks: pd.DataFrame, episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (training_table, flood_events) para los episodios históricos."""
    historical = episodes.loc[episodes["kind"] == "historical", ["episode_id", "started_at", "ended_at"]]
    static = zones[["zone_id", "altitude_avg", "slope_avg", "imperviousness", "patricova_score", "station_dist_m"]]
    table = peaks.merge(historical, on="episode_id").merge(static, on="zone_id")

    observed = sources.load_observed_floods(zones, episodes)
    if observed is None:
        table["severity"] = sample.flood_labels(table)
    else:
        table = table.merge(observed, on=["zone_id", "episode_id"], how="left")
    table["flooded"] = table["severity"].notna().astype(int)

    flooded = table[table["flooded"] == 1]
    events = pd.DataFrame({
        "id": np.arange(1, len(flooded) + 1),
        "zone_id": flooded["zone_id"].to_numpy(),
        "episode_id": flooded["episode_id"].to_numpy(),
        "started_at": flooded["started_at"].to_numpy(),
        "ended_at": flooded["ended_at"].to_numpy(),
        "severity": flooded["severity"].to_numpy(),
        "source_id": list(sources.SOURCES).index("flood_events") + 1,
    })
    table = table[["zone_id", "episode_id", "started_at", *FEATURES, "flooded"]]
    return table, events


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Zonas con atributos físicos, PATRICOVA, barrio y estación más cercana
    zones, stations = build_zones()
    log.info("zonas: %d celdas", len(zones))

    # 2. Episodios y lluvia horaria (más la previsión de AEMET si hay API key)
    episodes = sources.load_episodes()
    station_rain = sources.load_station_rain(stations, episodes)
    if AEMET_API_KEY:
        try:
            episodes, station_rain = add_aemet_forecast(episodes, station_rain, stations)
        except Exception as exc:
            log.warning("no se pudo descargar la previsión de AEMET: %s", exc)
    station_rain = features.rain_accumulations(station_rain)
    weather = features.weather_by_zone(zones[["zone_id", "station_id"]], station_rain)
    log.info("observaciones: %d filas, %d episodios", len(weather), episodes["episode_id"].nunique())

    # 3. Etiquetas y tabla de entrenamiento con los máximos de cada episodio histórico
    training, events = build_labels(zones, features.episode_peaks(weather), episodes)
    log.info("entrenamiento: %d filas, %.1f %% inundadas", len(training), 100 * training["flooded"].mean())

    # 4. Comprobaciones de datos antes de escribir nada
    errors = (
        checks.check_zones(zones)
        + checks.check_weather(weather, zones, episodes)
        + checks.check_flood_events(events, zones, episodes)
        + checks.check_training(training, FEATURES)
    )
    if errors:
        raise ValueError("Fallan las comprobaciones de datos:\n" + "\n".join(errors))

    # 5. Escritura en Parquet / GeoParquet
    zones.to_parquet(ZONES_FILE)
    episodes.to_parquet(EPISODES_FILE)
    weather.assign(year=weather["observed_at"].dt.year).to_parquet(
        WEATHER_DIR, partition_cols=["year"], existing_data_behavior="delete_matching"
    )
    events.to_parquet(FLOOD_EVENTS_FILE)
    training.to_parquet(TRAINING_FILE)
    sources.data_sources_table(pd.Timestamp.now().floor("s")).to_csv(DATA_SOURCES_FILE, index=False)
    log.info("fuentes: %s", sources.status)


if __name__ == "__main__":
    main()
