"""Métricas, umbrales de riesgo y asignación de niveles."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from src.config import RISK_LEVELS, TARGET_RECALL


def choose_thresholds(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Umbrales elegidos en validación.

    medio: el más alto que mantiene Recall >= objetivo (a partir de aquí la celda cuenta como positiva).
    alto: el de mejor F1, si queda por encima de medio; si no, el punto medio entre medio y 1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    meets_recall = recall >= TARGET_RECALL
    medio = float(thresholds[meets_recall].max()) if meets_recall.any() else float(thresholds.min())
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    alto = float(thresholds[f1.argmax()])
    if alto <= medio:
        alto = (medio + 1) / 2
    # Se trunca para no dejar fuera, por redondeo, las celdas justo en el umbral
    return {"medio": np.floor(medio * 1e4) / 1e4, "alto": np.floor(alto * 1e4) / 1e4}


def risk_level(proba: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    bins = [-np.inf, thresholds["medio"], thresholds["alto"], np.inf]
    return pd.cut(proba, bins=bins, labels=RISK_LEVELS, right=False).astype(str)


def metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Recall y F1 (principales), precisión, ROC-AUC, PR-AUC y matriz de confusión al umbral dado."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) > 1
    return {
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if both_classes else None,
        "pr_auc": round(average_precision_score(y_true, proba), 4) if both_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def false_negatives_by_district(frame: pd.DataFrame, threshold: float) -> list[dict]:
    """Informe de equidad (Fase 1, 9.4): celdas inundadas no detectadas por distrito."""
    flooded = frame[frame["flooded"] == 1]
    report = (
        flooded.assign(missed=flooded["proba"] < threshold)
        .groupby("district")
        .agg(flooded_cells=("missed", "size"), false_negatives=("missed", "sum"))
        .reset_index()
    )
    report["fn_rate"] = (report["false_negatives"] / report["flooded_cells"]).round(3)
    return report.sort_values("fn_rate", ascending=False).to_dict("records")
