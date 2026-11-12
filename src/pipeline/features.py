"""Cálculo de variables derivadas."""

import numpy as np
import pandas as pd

from src.config import PATRICOVA_MAX_LEVEL


def patricova_score(level: pd.Series) -> pd.Series:
    """Invierte el nivel PATRICOVA (1 = máxima peligrosidad) a una escala 0-6 donde más es peor (0 = fuera)."""
    return np.where(level > 0, PATRICOVA_MAX_LEVEL + 1 - level, 0)


def area_unit(zones: pd.DataFrame, level: str) -> pd.Series:
    """Unidad territorial de cada celda: barrio o distrito dentro de València; el municipio fuera de ella."""
    column = {"barrio": "neighbourhood", "distrito": "district"}[level]
    return zones[column].fillna(zones["municipality"])
