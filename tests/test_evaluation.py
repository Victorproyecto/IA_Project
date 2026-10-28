import numpy as np

from src.config import TARGET_RECALL
from src.models.evaluation import choose_thresholds, metrics, risk_level


def test_medium_threshold_keeps_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.4 + rng.uniform(0, 0.6, 500), 0, 1)
    thresholds = choose_thresholds(y, proba)
    assert thresholds["medio"] < thresholds["alto"]
    assert metrics(y, proba, thresholds["medio"])["recall"] >= TARGET_RECALL


def test_risk_levels_follow_thresholds():
    levels = risk_level(np.array([0.1, 0.5, 0.9]), {"medio": 0.5, "alto": 0.8})
    assert list(levels) == ["Bajo", "Medio", "Alto"]


def test_confusion_matrix_counts():
    result = metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["recall"] == 0.5
