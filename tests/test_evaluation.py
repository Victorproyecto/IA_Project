import numpy as np

from src.config import ALTO_RECALL, TARGET_RECALL
from src.pipeline.evaluation import choose_thresholds, metrics, risk_level

# Esta funcion testea que el umbral medio elegido mantiene el recall objetivo (TARGET_RECALL) y que el umbral alto es más exigente que el medio.
def test_medium_threshold_keeps_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.4 + rng.uniform(0, 0.6, 500), 0, 1)
    thresholds = choose_thresholds(y, proba)
    assert thresholds["medio"] < thresholds["alto"]
    assert metrics(y, proba, thresholds["medio"])["recall"] >= TARGET_RECALL

# Esta funcion testea que el umbral alto elegido mantiene al menos el recall objetivo para alto (ALTO_RECALL) y que es más exigente que el umbral medio.
def test_high_threshold_keeps_half_of_the_flooded():
    """Alto concentra al menos la mitad de lo inundado y es más exigente que Medio."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(y * 0.4 + rng.uniform(0, 0.6, 2000), 0, 1)
    thresholds = choose_thresholds(y, proba)
    assert thresholds["medio"] < thresholds["alto"]
    assert ALTO_RECALL <= metrics(y, proba, thresholds["alto"])["recall"] < TARGET_RECALL

# Esta funcion testea que la función risk_level asigna correctamente los niveles de riesgo (Bajo, Medio, Alto) según los umbrales elegidos.
def test_risk_levels_follow_thresholds():
    levels = risk_level(np.array([0.1, 0.5, 0.9]), {"medio": 0.5, "alto": 0.8})
    assert list(levels) == ["Bajo", "Medio", "Alto"]

# Esta funcion testea que la función metrics calcula correctamente la matriz de confusión y el recall para un conjunto de etiquetas verdaderas y probabilidades predichas, dado un umbral de decisión.
def test_confusion_matrix_counts():
    result = metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["recall"] == 0.5
