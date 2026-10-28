"""Ejecuta pipeline y entrenamiento una vez con los datos de muestra en un directorio temporal."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# DATA_DIR debe fijarse antes de importar src.config
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dana_test_")
os.environ["AEMET_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def built_data():
    from src.models import train
    from src.pipeline import run

    run.main()
    train.main()
    return Path(os.environ["DATA_DIR"])
