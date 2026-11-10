"""Ejecuta pipeline y entrenamiento una vez con los datos reales de data/raw en un directorio temporal.

Los ficheros de data/raw no se versionan: si faltan, los tests que dependen de ellos se saltan.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REAL_RAW = ROOT / "data" / "raw"

# DATA_DIR debe fijarse antes de importar src.config
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dana_test_")
(Path(os.environ["DATA_DIR"]) / "raw").symlink_to(REAL_RAW)
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def built_data():
    from src.pipeline import train
    from src.pipeline import run, sources

    missing = [name for name in sources.FILES.values() if not (REAL_RAW / name).exists()]
    if missing:
        pytest.skip(f"faltan datos reales en data/raw: {', '.join(missing)}")
    run.main()
    train.main()
    return Path(os.environ["DATA_DIR"])
