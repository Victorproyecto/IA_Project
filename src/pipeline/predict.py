"""Predicción de susceptibilidad por celda."""

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import FEATURES, PREDICTIONS_DIR
from src.pipeline.evaluation import risk_level


def predict_all(model, thresholds: dict, model_version: str, zones: gpd.GeoDataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """FloodPrediction: una fila por celda con risk_probability y risk_level del modelo final.

    Se guardan también la probabilidad de validación cruzada (oof_probability) y lo observado en la DANA.
    """
    proba = model.predict_proba(zones[FEATURES])[:, 1]
    predictions = pd.DataFrame({
        "id": np.arange(1, len(zones) + 1),
        "zone_id": zones["zone_id"].to_numpy(),
        "created_at": pd.Timestamp.now().floor("s"),
        "risk_probability": proba.round(4),
        "risk_level": risk_level(proba, thresholds),
        "model_version": model_version,
    })
    observed = table[["zone_id", "proba", "flooded"]].rename(columns={"proba": "oof_probability"})
    return predictions.merge(observed.round({"oof_probability": 4}), on="zone_id", how="left")


def save(predictions: pd.DataFrame, model_version: str) -> None:
    """Guarda las predicciones en data/procesados/predictions/model_version=<versión>/ (una carpeta por versión del
    modelo)."""
    target = PREDICTIONS_DIR / f"model_version={model_version}"
    target.mkdir(parents=True, exist_ok=True)
    predictions.drop(columns="model_version").to_parquet(target / "predictions.parquet")
