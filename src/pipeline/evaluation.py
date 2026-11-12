"""Métricas, umbrales de riesgo y asignación de niveles."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from src.config import ALTO_RECALL, RISK_LEVELS, TARGET_RECALL

# Los umbrales se guardan con 4 decimales, truncados para no dejar fuera por redondeo las celdas justo en el umbral
THRESHOLD_DECIMALS = 4

# Esta función elige los umbrales de riesgo medio y alto a partir de las probabilidades de predicción y las etiquetas verdaderas. Devuelve un diccionario con los umbrales elegidos.
def choose_thresholds(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    # Umbrales elegidos en validación.

    # medio: el más alto que mantiene Recall >= TARGET_RECALL: Medio + Alto encuentran el 80 % de lo inundado.
    # alto: el más alto que mantiene Recall >= ALTO_RECALL: Alto concentra la mitad de lo inundado, la de mayor
    # probabilidad. No depende de cuánto se inundó el área. Si quedara por debajo de medio, el punto medio entre medio y 1.
    
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    recall = recall[:-1]
    meets_recall = recall >= TARGET_RECALL
    medio = float(thresholds[meets_recall].max()) if meets_recall.any() else float(thresholds.min())
    meets_alto = recall >= ALTO_RECALL
    alto = float(thresholds[meets_alto].max()) if meets_alto.any() else (medio + 1) / 2
    if alto <= medio:
        alto = (medio + 1) / 2
    scale = 10**THRESHOLD_DECIMALS
    umbrales = {"medio": np.floor(medio * scale) / scale, "alto": np.floor(alto * scale) / scale}
    return umbrales

# Esta función asigna un nivel de riesgo (Bajo, Medio o Alto) a cada probabilidad de predicción según los umbrales elegidos. Devuelve un array con los niveles de riesgo correspondientes.
def risk_level(proba: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    # Nivel Bajo, Medio o Alto de cada probabilidad (una probabilidad igual al umbral entra en el nivel superior).
    bins = [-np.inf, thresholds["medio"], thresholds["alto"], np.inf]
    return pd.cut(proba, bins=bins, labels=RISK_LEVELS, right=False).astype(str)

# Esta función calcula varias métricas de evaluación (recall, F1, precisión, ROC-AUC, PR-AUC y matriz de confusión) para un conjunto de etiquetas verdaderas y probabilidades predichas, dado un umbral de decisión. Devuelve un diccionario con las métricas calculadas.
def metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    # Recall y F1 (principales), precisión, ROC-AUC, PR-AUC y matriz de confusión al umbral dado.
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) > 1
    metricas = {
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if both_classes else None,
        "pr_auc": round(average_precision_score(y_true, proba), 4) if both_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metricas

# Esta función realiza una comprobación con calados medidos en campo, comparando los puntos de profundidad con las predicciones del modelo y los mapas oficiales. Devuelve un diccionario con el porcentaje de detección del modelo y de los mapas oficiales.
def field_depth_detection(depth_m: pd.Series, flagged: pd.Series, min_depth: float) -> dict:
    # Comprobación con calados de campo: % de puntos con calado medido >= min_depth que caen en celdas marcadas.
    wet = depth_m >= min_depth
    return {
        "points": int(wet.sum()),
        "detected_pct": round(100 * float(flagged[wet].mean()), 1) if wet.any() else None,
    }

# Esta función calcula el número de celdas inundadas no detectadas por distrito, dado un DataFrame con las etiquetas verdaderas y las probabilidades predichas, y un umbral de decisión. Devuelve una lista de diccionarios con los resultados por distrito.
def false_negatives_by_district(frame: pd.DataFrame, threshold: float) -> list[dict]:
    # Informe de equidad (Fase 1, 9.4): celdas inundadas no detectadas por distrito.
    flooded = frame[frame["flooded"] == 1]
    report = (
        flooded.assign(missed=flooded["proba"] < threshold)
        .groupby("district")
        .agg(flooded_cells=("missed", "size"), false_negatives=("missed", "sum"))
        .reset_index()
    )
    report["fn_rate"] = (report["false_negatives"] / report["flooded_cells"]).round(3)
    reporte = report.sort_values("fn_rate", ascending=False).to_dict("records")
    return reporte
