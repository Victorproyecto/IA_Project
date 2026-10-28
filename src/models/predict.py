"""Predicción horaria por celda para todos los episodios, escenarios y previsiones."""

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import FEATURES, PREDICTIONS_DIR, WEATHER_DIR
from src.models.evaluation import risk_level


def predict_all(model, thresholds: dict, model_version: str, zones: gpd.GeoDataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """FloodPrediction: una fila por celda, episodio y hora con risk_probability y risk_level."""
    weather = pd.read_parquet(WEATHER_DIR, columns=["zone_id", "episode_id", "observed_at", "precipitation_1h", "precipitation_24h"])
    static = zones[["zone_id", "altitude_avg", "slope_avg", "imperviousness", "patricova_score", "station_dist_m"]]
    frame = weather.merge(static, on="zone_id")

    proba = model.predict_proba(frame[FEATURES])[:, 1]
    predictions = pd.DataFrame({
        "id": np.arange(1, len(frame) + 1),
        "zone_id": frame["zone_id"].to_numpy(),
        "episode_id": frame["episode_id"].to_numpy(),
        "created_at": pd.Timestamp.now().floor("s"),
        "target_time": frame["observed_at"].to_numpy(),
        "risk_probability": proba.round(4),
        "risk_level": risk_level(proba, thresholds),
        "model_version": model_version,
    })
    simulated = episodes.set_index("episode_id")["is_simulated"]
    predictions["is_simulated"] = predictions["episode_id"].map(simulated).astype(bool)
    return predictions


def save(predictions: pd.DataFrame, model_version: str) -> None:
    target = PREDICTIONS_DIR / f"model_version={model_version}"
    target.mkdir(parents=True, exist_ok=True)
    predictions.drop(columns="model_version").to_parquet(target / "predictions.parquet")
