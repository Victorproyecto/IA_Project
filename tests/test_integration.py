"""Integración: Streamlit carga los Parquet generados y responde a los controles."""

from streamlit.testing.v1 import AppTest

APP = "../src/app/streamlit_app.py"


def test_app_loads_with_disclaimer(built_data):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("no sustituye los avisos de las autoridades" in w.value for w in at.warning)


def test_app_hour_slider_and_scenario(built_data):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.checkbox[0].uncheck().run()
    assert not at.exception
    scenario = next(o for o in at.sidebar.selectbox[0].options if "Escenario" in o)
    at.sidebar.selectbox[0].select(scenario).run()
    assert not at.exception
    assert any("ESCENARIO SIMULADO" in m.value for m in at.markdown)
