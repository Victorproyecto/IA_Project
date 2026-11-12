"""Integración: Streamlit carga los Parquet generados y muestra el aviso y la explicación de las capas."""

from streamlit.testing.v1 import AppTest

APP = "../src/frontend/app.py"
TIMEOUT_S = 60


def test_app_loads_with_disclaimer_and_layer_help(built_data):
    """Arranca la app una sola vez: aviso de que no sustituye a las autoridades y ayuda del control de capas (celdas y
    lo inundado en la DANA)."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    assert not at.exception
    assert any("no sustituye los avisos de las autoridades" in w.value for w in at.warning)
    assert any("control de capas" in c.value and "Copernicus EMS" in c.value for c in at.caption)
