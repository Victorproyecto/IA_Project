"""Componentes JavaScript propios de la aplicación (st.components.v2): el mapa de celdas y el buscador con sugerencias.

Cada componente tiene su JavaScript y su CSS en esta carpeta; aquí se registran y se les pasan los datos.
"""

from collections.abc import Callable
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent


def _read(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


# El mapa no aísla sus estilos: MapLibre añade su CSS a la página
_cell_map = st.components.v2.component(
    "dana_cell_map", js=_read("cell_map.js"), css=_read("cell_map.css"), isolate_styles=False
)
_place_search = st.components.v2.component("dana_place_search", js=_read("place_search.js"), css=_read("place_search.css"))
_theme_toggle = st.components.v2.component("dana_theme_toggle", js=_read("theme_toggle.js"),
                                           css=_read("theme_toggle.css"))


def theme_toggle(*, key: str):
    """Botón para pasar de tema claro a oscuro y al revés. Guarda la elección como lo hace el menú de ajustes de
    Streamlit y recarga la página."""
    return _theme_toggle(data={}, key=key)


def cell_map(*, grid: dict, colors: dict, opacity: dict, observed_colors: dict, observed_opacity: dict, area: dict,
             dark: bool, selected: int | None, view: dict, height: int, key: str, on_cell: Callable[[], None]):
    """Mapa de celdas. `area` es el límite del área piloto (GeoJSON); con `dark`, el mapa de fondo se oscurece.
    `on_cell` se ejecuta al pulsar el mapa, antes de volver a ejecutar la página; el zone_id pulsado está en
    st.session_state[key]["cell"] (0 si se pulsa fuera de las celdas)."""
    data = {"grid": grid, "colors": colors, "opacity": opacity, "observed_colors": observed_colors,
            "observed_opacity": observed_opacity, "area": area, "dark": dark, "selected": selected, "view": view,
            "height": height}
    return _cell_map(data=data, key=key, height=height, on_cell_change=on_cell)


def place_search(*, places: list[dict], postal_codes: list[str], label: str, placeholder: str, key: str,
                 on_place: Callable[[], None]):
    """Buscador con sugerencias. `on_place` se ejecuta al elegir una; el lugar elegido ({lat, lon, label, kind})
    está en st.session_state[key]["place"]."""
    data = {"places": places, "postal_codes": postal_codes, "label": label, "placeholder": placeholder}
    return _place_search(data=data, key=key, on_place_change=on_place)
