import json

from src import config


def test_model_card_reports_metrics_and_criteria(built_data):
    card = json.loads(config.MODEL_CARD_FILE.read_text(encoding="utf-8"))
    assert card["chosen_model"] in card["models"]
    assert {"recall", "f1", "roc_auc", "confusion_matrix"} <= set(card["models"][card["chosen_model"]]["validation"])
    assert len(card["models"][card["chosen_model"]]["roc_auc_by_fold"]) == config.N_FOLDS
    assert 0 < card["thresholds"]["medio"] < card["thresholds"]["alto"] < 1
    assert "beats_references" in card["success_criteria"]


def test_model_card_reports_references_and_checks(built_data):
    card = json.loads(config.MODEL_CARD_FILE.read_text(encoding="utf-8"))
    assert set(card["models"]) == {"logistic_regression", "random_forest"}
    assert set(card["feature_importance"]) == set(config.FEATURES)
    # Los mapas oficiales no son variables del modelo: solo referencia y una prueba aparte "con PATRICOVA"
    assert "patricova_score" not in config.FEATURES and not any(f.startswith("snczi") for f in config.FEATURES)
    assert "patricova_score" in card["ablation_with_patricova"]["features"]
    assert set(card["baselines"]) == {"PATRICOVA", "SNCZI T=500"}
    assert card["field_check"]["points"] > 0
    assert 0 <= card["field_check"]["model_detected_pct"] <= 100
    assert 0 < card["mark_all_f1"] < 1


def test_external_test_on_unseen_basin(built_data):
    card = json.loads(config.MODEL_CARD_FILE.read_text(encoding="utf-8"))
    external = card["external_test"]
    assert external["region"] == "La Ribera" and external["test_cells"] > 0
    assert external["gap_m"] == config.GENERALIZATION_GAP_M
    assert {"f1", "roc_auc"} <= set(external["model"])
    assert set(external["references"]) == {"PATRICOVA", "SNCZI T=500"}
