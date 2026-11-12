"""Ejecuta el pipeline y un entrenamiento rápido una vez, con los datos reales de data/preparados, en un directorio
temporal y sobre un recorte pequeño del área.

- Directorio temporal: los tests no tocan data/procesados ni data/modelos.
- Recorte: 6 x 6 km entre Paiporta, Picanya, Catarroja y el sur de València (~3.500 celdas, con barrios y distritos)
  y una ventana de la Ribera (Algemesí, Sollana) como prueba externa. El área completa tarda ~5 min en preparar;
  el recorte, segundos. Los tests comprueban que el proceso funciona y que los datos son coherentes, no las métricas
  del modelo final (esas están en data/modelos/model_card.json).
- Si falta algún fichero preparado, los tests que dependen de ellos se saltan.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PREPARED = ROOT / "data" / "preparados"
# Los originales (data/raw) no se versionan: si están (en local), se enlazan para el test de la huella de la GVA
RAW = ROOT / "data" / "raw"


def _link_dir(link: Path, target: Path) -> None:
    """Enlaza data/preparados sin copiarlo. En Windows sin modo desarrollador no se pueden crear enlaces simbólicos:
    se usa una unión de directorio (junction), que no necesita permisos especiales."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


# DATA_DIR debe fijarse antes de importar src.config
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dana_test_")
# Sin data/preparados (antes de la ingesta, o en la CI) no se enlaza nada y los tests con datos se saltan
if PREPARED.exists():
    _link_dir(Path(os.environ["DATA_DIR"]) / "preparados", PREPARED)
if RAW.exists():
    _link_dir(Path(os.environ["DATA_DIR"]) / "raw", RAW)
sys.path.insert(0, str(ROOT))

# El recorte se fija antes de que los módulos del pipeline importen las áreas de config (UTM 30N, múltiplos de 100 m)
from src import config  # noqa: E402

config.PILOT_EXTENT = (720000, 4364000, 726000, 4370000)
config.EXTERNAL_TEST_EXTENT = (722000, 4343000, 728000, 4349000)


@pytest.fixture(scope="session")
def built_data():
    from src.config import PREPARED_FILES
    from src.pipeline import run, train

    missing = [name for name in PREPARED_FILES.values() if not (PREPARED / name).exists()]
    if missing:
        pytest.skip(f"faltan datos en data/preparados: {', '.join(missing)}")
    run.main()
    train.main(quick=True)
    return Path(os.environ["DATA_DIR"])
