import json

import numpy as np
import pandas as pd

from src import config
from src.models.train import candidates, temporal_split


def test_models_return_probabilities_on_small_dataset(built_data):
    table = pd.read_parquet(config.TRAINING_FILE).sample(2000, random_state=0)
    for model in candidates().values():
        model.fit(table[config.FEATURES], table["flooded"])
        proba = model.predict_proba(table[config.FEATURES])[:, 1]
        assert np.all((proba >= 0) & (proba <= 1))


def test_retrospective_episode_is_excluded_from_training(built_data):
    split = temporal_split(pd.read_parquet(config.EPISODES_FILE))
    assert not set(split["test"]) & set(split["train"])
    assert not set(split["validation"]) & set(split["train"])


def test_model_card_reports_metrics_and_criteria(built_data):
    card = json.loads(config.MODEL_CARD_FILE.read_text())
    assert card["chosen_model"] in card["models"]
    assert {"recall", "f1", "roc_auc", "confusion_matrix"} <= set(card["models"][card["chosen_model"]]["validation"])
    assert 0 < card["thresholds"]["medio"] < card["thresholds"]["alto"] < 1
    assert "beats_references" in card["success_criteria"]
    assert card["trained_on_simulated_data"] is True
