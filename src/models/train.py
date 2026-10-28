"""Entrenamiento, validación temporal, prueba retrospectiva y predicciones. Uso: python -m src.models.train"""

import json
import logging

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_SOURCES_FILE, EPISODES_FILE, FEATURES, MODEL_CARD_FILE, MODELS_DIR, RANDOM_STATE, TARGET_F1,
    TARGET_RECALL, TRAINING_FILE, ZONES_FILE,
)
from src.models import predict
from src.models.evaluation import choose_thresholds, false_negatives_by_district, metrics

log = logging.getLogger("train")

VERSION = "0.1.0"
VALIDATION_EPISODES = 3


def candidates() -> dict:
    """Pesos de clase por el desequilibrio y calibración para que risk_probability sea interpretable."""
    base = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    return {name: CalibratedClassifierCV(model, method="sigmoid", cv=3) for name, model in base.items()}


def feature_importance(model, chosen: str) -> dict[str, float]:
    """Importancia (RF) o coeficiente estandarizado (LR), promediado entre los modelos calibrados."""
    estimators = [c.estimator for c in model.calibrated_classifiers_]
    if chosen == "random_forest":
        values = np.mean([e.feature_importances_ for e in estimators], axis=0)
    else:
        values = np.mean([e[-1].coef_[0] for e in estimators], axis=0)
    return dict(zip(FEATURES, values.round(4).tolist()))


def temporal_split(episodes: pd.DataFrame) -> dict[str, list[int]]:
    """El último episodio histórico es la prueba retrospectiva; los anteriores, validación; el resto, entrenamiento."""
    historical = episodes[episodes["kind"] == "historical"].sort_values("started_at")["episode_id"].tolist()
    return {
        "train": historical[: -VALIDATION_EPISODES - 1],
        "validation": historical[-VALIDATION_EPISODES - 1 : -1],
        "test": historical[-1:],
    }


def baselines(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Referencias a superar: umbral de lluvia acumulada en 24 h y peligrosidad PATRICOVA."""
    results = {}
    for name, column in [("rain_threshold_24h", "precipitation_24h"), ("patricova_reference", "patricova_score")]:
        scale = train[column].max() or 1
        thresholds = choose_thresholds(validation["flooded"], validation[column] / scale)
        results[name] = {
            "threshold": round(thresholds["medio"] * scale, 3),
            "validation": metrics(validation["flooded"], validation[column] / scale, thresholds["medio"]),
            "test": metrics(test["flooded"], test[column] / scale, thresholds["medio"]),
        }
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(TRAINING_FILE)
    episodes = pd.read_parquet(EPISODES_FILE)
    zones = gpd.read_parquet(ZONES_FILE)
    table = table.merge(zones[["zone_id", "district"]], on="zone_id")

    # 1. División temporal por episodio
    split = temporal_split(episodes)
    train, validation, test = (table[table["episode_id"].isin(split[k])] for k in ("train", "validation", "test"))
    log.info("train %d, validation %d, test %d filas", len(train), len(validation), len(test))

    # 2. Entrenamiento y umbrales en validación con los mismos conjuntos para todos los modelos
    results, fitted = {}, {}
    for name, model in candidates().items():
        model.fit(train[FEATURES], train["flooded"])
        val_proba = model.predict_proba(validation[FEATURES])[:, 1]
        thresholds = choose_thresholds(validation["flooded"].to_numpy(), val_proba)
        results[name] = {
            "params": {k: str(v) for k, v in model.estimator.get_params().items() if "__" not in k},
            "thresholds": thresholds,
            "validation": metrics(validation["flooded"], val_proba, thresholds["medio"]),
        }
        fitted[name] = model
        log.info("%s validación: %s", name, results[name]["validation"])

    # 3. Selección: el candidato solo se elige si supera al modelo base en Recall y F1
    base, cand = results["logistic_regression"]["validation"], results["random_forest"]["validation"]
    chosen = "random_forest" if cand["recall"] >= base["recall"] and cand["f1"] > base["f1"] else "logistic_regression"
    model, thresholds = fitted[chosen], results[chosen]["thresholds"]

    # 4. Prueba retrospectiva con el episodio excluido del entrenamiento
    test = test.assign(proba=model.predict_proba(test[FEATURES])[:, 1])
    retrospective = metrics(test["flooded"], test["proba"].to_numpy(), thresholds["medio"])
    for name in results:
        test_proba = fitted[name].predict_proba(test[FEATURES])[:, 1]
        results[name]["test"] = metrics(test["flooded"], test_proba, results[name]["thresholds"]["medio"])

    reference = baselines(train, validation, test)
    chosen_val = results[chosen]["validation"]
    # Todos operan con Recall >= objetivo, así que se comparan por F1 en ese punto
    beats_references = chosen_val["recall"] >= TARGET_RECALL and all(
        chosen_val["f1"] > ref["validation"]["f1"] for ref in reference.values()
    )

    model_version = f"{VERSION}-{'rf' if chosen == 'random_forest' else 'lr'}"
    importances = feature_importance(model, chosen)
    data_status = pd.read_csv(DATA_SOURCES_FILE).set_index("name")["status"].to_dict()

    card = {
        "model_version": model_version,
        "chosen_model": chosen,
        "trained_at": pd.Timestamp.now().floor("s").isoformat(),
        "features": FEATURES,
        "feature_importance": importances,
        "thresholds": thresholds,
        "split": {k: episodes.set_index("episode_id").loc[v, "name"].tolist() for k, v in split.items()},
        "models": results,
        "baselines": reference,
        "retrospective_test": {
            "episode": episodes.set_index("episode_id").loc[split["test"][0], "name"],
            "metrics": retrospective,
            "false_negatives_by_district": false_negatives_by_district(test, thresholds["medio"]),
        },
        "success_criteria": {
            "recall_target": TARGET_RECALL,
            "f1_target": TARGET_F1,
            "validation_recall_ok": chosen_val["recall"] >= TARGET_RECALL,
            "validation_f1_ok": chosen_val["f1"] >= TARGET_F1,
            "candidate_beats_baseline": chosen == "random_forest",
            "beats_references": beats_references,
        },
        "data_sources": data_status,
        "trained_on_simulated_data": "simulado" in data_status.values(),
    }

    # 5. Guardado del modelo, la ficha, el registro de experimentos y las predicciones
    joblib.dump(model, MODELS_DIR / f"model_{model_version}.joblib")
    MODEL_CARD_FILE.write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str))
    with open(MODELS_DIR / "experiments.jsonl", "a") as f:
        f.write(json.dumps({k: card[k] for k in ("model_version", "trained_at", "split", "models", "baselines")}, default=str) + "\n")

    predictions = predict.predict_all(model, thresholds, model_version, zones, episodes)
    predict.save(predictions, model_version)
    log.info("modelo elegido %s, retrospectiva %s", model_version, retrospective)
    log.info("predicciones: %d filas, niveles %s", len(predictions), predictions["risk_level"].value_counts().to_dict())


if __name__ == "__main__":
    main()
