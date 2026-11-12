"""Entrenamiento con validación cruzada por bloques espaciales y predicciones. Uso: python -m src.pipeline.train"""

import json
import logging

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BLOCK_SIZE_CELLS, CELL_SIZE_M, DATA_SOURCES_FILE, EPISODE_NAME, EXPERIMENTS_FILE, EXTERNAL_TEST_FILE, FEATURES,
    FIELD_DEPTH_MIN_M, GENERALIZATION_GAP_M, MODEL_CARD_FILE, MODELS_DIR, N_FOLDS, PATRICOVA_MAX_LEVEL, PILOT_EXTENT,
    RANDOM_STATE, SNCZI_COLUMN, TARGET_F1, TARGET_RECALL, TRAINING_FILE, ZONES_FILE,
)
from src.pipeline import predict, sources
from src.pipeline.evaluation import (
    choose_thresholds, false_negatives_by_district, field_depth_detection, metrics,
)
from src.pipeline.features import area_unit

log = logging.getLogger("train")

VERSION = "0.3.0"
SHORT_NAMES = {"logistic_regression": "lr", "random_forest": "rf"}

# Hiperparámetros. El bosque usa hojas de al menos 5 celdas para no memorizar celdas sueltas
RF_TREES = 300
RF_MIN_SAMPLES_LEAF = 5
LR_MAX_ITER = 1000
# Particiones internas de la calibración sigmoide (sobre los datos de entrenamiento de cada modelo)
CALIBRATION_FOLDS = 3
# Repeticiones de la importancia por permutación (cada variable se desordena estas veces y se promedia)
PERMUTATION_REPEATS = 5
# Ejecución rápida para los tests: comprueba que todo funciona de principio a fin, no la calidad del modelo
QUICK_RF_TREES = 10
QUICK_PERMUTATION_REPEATS = 1

# Mapas oficiales de peligrosidad: no son variables del modelo, son la referencia con la que se compara.
# Nombre -> (columna de zones, valor máximo para pasarla a 0-1)
REFERENCES = {"PATRICOVA": ("patricova_score", PATRICOVA_MAX_LEVEL), "SNCZI T=500": (SNCZI_COLUMN, 1)}
REFERENCE_COLUMNS = [column for column, _ in REFERENCES.values()]
# Un mapa oficial marca "en riesgo" cualquier celda dentro de una de sus zonas (puntuación > 0)
IN_HAZARD_ZONE = 1e-9


def candidates(rf_trees: int = RF_TREES) -> dict:
    """Pesos de clase por el desequilibrio y calibración para que risk_probability sea interpretable."""
    base = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=LR_MAX_ITER)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_trees, min_samples_leaf=RF_MIN_SAMPLES_LEAF, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    return {name: CalibratedClassifierCV(model, method="sigmoid", cv=CALIBRATION_FOLDS) for name, model in base.items()}


def feature_importance(model, table: pd.DataFrame, repeats: int = PERMUTATION_REPEATS) -> dict[str, float]:
    """Importancia por permutación: cuánto cae la PR-AUC al desordenar cada variable. Mismo criterio para los dos
    modelos (los coeficientes de la regresión y las importancias del bosque no son comparables entre sí)."""
    result = permutation_importance(
        model, table[FEATURES], table["flooded"], scoring="average_precision", n_repeats=repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return dict(zip(FEATURES, result.importances_mean.round(4).tolist(), strict=True))


def out_of_fold(model, table: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    """Probabilidad de cada celda predicha por un modelo entrenado sin el grupo de bloques de esa celda."""
    proba = np.zeros(len(table))
    for fold in range(N_FOLDS):
        held_out = (table["fold"] == fold).to_numpy()
        model.fit(table.loc[~held_out, features], table.loc[~held_out, "flooded"])
        proba[held_out] = model.predict_proba(table.loc[held_out, features])[:, 1]
    return proba


def roc_auc_by_fold(table: pd.DataFrame, proba: np.ndarray) -> list[float]:
    """ROC-AUC de cada grupo por separado, para ver cuánto varía según la zona."""
    return [round(roc_auc_score(g["flooded"], proba[g.index]), 4) for _, g in table.groupby("fold")]


def mark_all_f1(flooded: pd.Series) -> float:
    """F1 de marcarlo todo como inundado: con muchas celdas inundadas, el F1 sube solo; es la referencia trivial."""
    share = float(flooded.mean())
    return round(2 * share / (1 + share), 3)


def official_references(table: pd.DataFrame) -> dict:
    """Cada mapa oficial tal cual: una celda en su zona de peligrosidad cuenta como "en riesgo"."""
    result = {}
    for name, (column, scale) in REFERENCES.items():
        score = table[column] / scale
        result[name] = {"column": column, "validation": metrics(table["flooded"], score, IN_HAZARD_ZONE)}
    return result


def with_patricova(chosen: str, table: pd.DataFrame, rf_trees: int = RF_TREES) -> dict:
    """Prueba: el modelo elegido con PATRICOVA como variable más. Muestra cuánto añadiría un mapa oficial."""
    features = [*FEATURES, "patricova_score"]
    proba = out_of_fold(candidates(rf_trees)[chosen], table, features)
    threshold = choose_thresholds(table["flooded"].to_numpy(), proba)["medio"]
    return {"features": features, "validation": metrics(table["flooded"], proba, threshold)}


def _scores(test: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict:
    """Modelo y mapas oficiales sobre las mismas celdas de prueba, más la referencia trivial."""
    return {
        "test_cells": int(len(test)),
        "test_flooded_share": round(float(test["flooded"].mean()), 3),
        "mark_all_f1": mark_all_f1(test["flooded"]),
        "model": metrics(test["flooded"], proba, threshold),
        "references": {name: metrics(test["flooded"], test[column] / scale, IN_HAZARD_ZONE)
                       for name, (column, scale) in REFERENCES.items()},
    }


def external_test(model, threshold: float) -> dict:
    """Prueba externa en la Ribera, otra cuenca (Magro y Xúquer) que el modelo nunca ha visto: el modelo final,
    entrenado con todo el área piloto, predice sus celdas a más de GENERALIZATION_GAP_M del área piloto."""
    external = gpd.read_parquet(EXTERNAL_TEST_FILE)
    far = external.geometry.centroid.y < PILOT_EXTENT[1] - GENERALIZATION_GAP_M
    external = external[far.to_numpy()]
    return {"region": "La Ribera", "gap_m": GENERALIZATION_GAP_M,
            **_scores(external, model.predict_proba(external[FEATURES])[:, 1], threshold)}


def field_check(zones: gpd.GeoDataFrame, validated: pd.DataFrame, threshold: float) -> dict:
    """Comprobación con calados medidos en campo: % de puntos con agua en celdas Medio/Alto, modelo y mapas oficiales.

    No es una validación independiente: la huella de la GVA, que forma parte de la etiqueta, cubre casi todos estos
    puntos. Sirve para comprobar dónde había agua de verdad y comparar con los mapas oficiales.
    """
    points = gpd.sjoin(sources.load_field_depths(), zones[["zone_id", "geometry"]], predicate="within")
    points = points.merge(validated[["zone_id", "proba", *REFERENCE_COLUMNS]], on="zone_id")
    model = field_depth_detection(points["depth_m"], points["proba"] >= threshold, FIELD_DEPTH_MIN_M)
    references = {name: field_depth_detection(points["depth_m"], points[column] > 0, FIELD_DEPTH_MIN_M)["detected_pct"]
                  for name, (column, _) in REFERENCES.items()}
    return {
        "min_depth_m": FIELD_DEPTH_MIN_M,
        "points": model["points"],
        "cells_with_points": int(points.loc[points["depth_m"] >= FIELD_DEPTH_MIN_M, "zone_id"].nunique()),
        "model_detected_pct": model["detected_pct"],
        "references_detected_pct": references,
    }


def evaluate_candidates(table: pd.DataFrame, rf_trees: int = RF_TREES) -> tuple[dict, dict]:
    """Validación cruzada por bloques de cada candidato: cada celda recibe la probabilidad de un modelo que no vio su
    zona. Devuelve (resultados por modelo, probabilidades de validación por modelo)."""
    results, oof = {}, {}
    for name, model in candidates(rf_trees).items():
        oof[name] = out_of_fold(model, table)
        thresholds = choose_thresholds(table["flooded"].to_numpy(), oof[name])
        results[name] = {
            "params": {k: str(v) for k, v in model.estimator.get_params().items() if "__" not in k},
            "thresholds": thresholds,
            "validation": metrics(table["flooded"], oof[name], thresholds["medio"]),
            "roc_auc_by_fold": roc_auc_by_fold(table, oof[name]),
        }
        log.info("%s validación: %s", name, results[name]["validation"])
    return results, oof


def select_model(results: dict) -> str:
    """Entre los modelos que alcanzan el Recall objetivo, el de mayor F1 (desempate por PR-AUC)."""
    eligible = [n for n, r in results.items() if r["validation"]["recall"] >= TARGET_RECALL] or list(results)
    return max(eligible, key=lambda n: (results[n]["validation"]["f1"], results[n]["validation"]["pr_auc"]))


def _f1_and_roc(result: dict) -> dict:
    return {k: result[k] for k in ("f1", "roc_auc")}


def log_summary(card: dict, predictions: pd.DataFrame) -> None:
    """Resumen de la ficha en el log: modelo, referencias oficiales y pruebas en zonas no vistas."""
    log.info("modelo elegido %s, validación %s (F1 de marcarlo todo: %s)",
             card["model_version"], card["models"][card["chosen_model"]]["validation"], card["mark_all_f1"])
    log.info("mapas oficiales tal cual: %s", {n: _f1_and_roc(r["validation"]) for n, r in card["baselines"].items()})
    log.info("con PATRICOVA como variable: %s", _f1_and_roc(card["ablation_with_patricova"]["validation"]))
    external = card["external_test"]
    log.info("prueba externa (la Ribera): modelo %s | %s", _f1_and_roc(external["model"]),
             {n: _f1_and_roc(m) for n, m in external["references"].items()})
    log.info("comprobación con calados de campo: %s", card["field_check"])
    log.info("predicciones: %d celdas, niveles %s", len(predictions), predictions["risk_level"].value_counts().to_dict())


def main(quick: bool = False) -> None:
    """Entrena, valida y guarda modelo, ficha y predicciones. quick=True (tests): bosque pequeño y una sola repetición
    de la importancia por permutación; mismo proceso completo, sin buscar la calidad del modelo."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rf_trees = QUICK_RF_TREES if quick else RF_TREES
    repeats = QUICK_PERMUTATION_REPEATS if quick else PERMUTATION_REPEATS

    table = pd.read_parquet(TRAINING_FILE)
    zones = gpd.read_parquet(ZONES_FILE)
    table = table.merge(zones[["zone_id", "municipality", "district", "neighbourhood", *REFERENCE_COLUMNS]], on="zone_id")
    table = table.reset_index(drop=True)
    log.info("%d celdas en %d grupos de bloques", len(table), N_FOLDS)

    # Validación cruzada por bloques de cada candidato y selección
    results, oof = evaluate_candidates(table, rf_trees)
    chosen = select_model(results)
    thresholds = results[chosen]["thresholds"]

    # Modelo final entrenado con todas las celdas
    model = candidates(rf_trees)[chosen].fit(table[FEATURES], table["flooded"])

    references = official_references(table)
    chosen_val = results[chosen]["validation"]
    # Se compara con el mejor mapa oficial tal cual, por F1
    best_reference_f1 = max(r["validation"]["f1"] for r in references.values())
    beats_reference = chosen_val["recall"] >= TARGET_RECALL and chosen_val["f1"] > best_reference_f1
    validated = table.assign(proba=oof[chosen], district=area_unit(table, "distrito"))

    model_version = f"{VERSION}-{SHORT_NAMES[chosen]}"
    card = {
        "model_version": model_version,
        "chosen_model": chosen,
        "trained_at": pd.Timestamp.now().floor("s").isoformat(),
        "target": f"Celda inundada en {EPISODE_NAME} (Copernicus EMS EMSR773 + huella oficial de la GVA)",
        "area": "València, L'Horta Sud y L'Horta Oest (la Ribera solo como prueba externa)",
        "cell_size_m": CELL_SIZE_M,
        "features": FEATURES,
        "feature_importance": feature_importance(model, table, repeats),
        "feature_importance_method": "permutación: caída de PR-AUC al desordenar la variable",
        "thresholds": thresholds,
        "validation_method": (
            f"validación cruzada en {N_FOLDS} grupos de bloques espaciales de {BLOCK_SIZE_CELLS * CELL_SIZE_M} m; "
            "umbrales fijados sobre esas mismas predicciones"
        ),
        "selection_rule": "entre los modelos con Recall >= objetivo, el de mayor F1 (desempate por PR-AUC)",
        "cells": int(len(table)),
        "flooded_cells": int(table["flooded"].sum()),
        "mark_all_f1": mark_all_f1(table["flooded"]),
        "models": results,
        "baselines": references,
        "ablation_with_patricova": with_patricova(chosen, table, rf_trees),
        "external_test": external_test(model, thresholds["medio"]),
        "field_check": field_check(zones, validated, thresholds["medio"]),
        "false_negatives_by_district": false_negatives_by_district(validated, thresholds["medio"]),
        "success_criteria": {
            "recall_target": TARGET_RECALL,
            "f1_target": TARGET_F1,
            "validation_recall_ok": chosen_val["recall"] >= TARGET_RECALL,
            "validation_f1_ok": chosen_val["f1"] >= TARGET_F1,
            "candidate_beats_baseline": chosen != "logistic_regression",
            "beats_references": beats_reference,
        },
        "data_sources": pd.read_csv(DATA_SOURCES_FILE)["name"].tolist(),
    }

    # Guardado del modelo, la ficha, el registro de experimentos y las predicciones
    joblib.dump(model, MODELS_DIR / f"model_{model_version}.joblib")
    MODEL_CARD_FILE.write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with open(EXPERIMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: card[k] for k in ("model_version", "trained_at", "validation_method", "models", "baselines")},
                           ensure_ascii=False, default=str) + "\n")

    predictions = predict.predict_all(model, thresholds, model_version, zones, validated)
    predict.save(predictions, model_version)
    log_summary(card, predictions)


if __name__ == "__main__":
    main()
