"""Frontend DANA Predictor (Streamlit). Uso: streamlit run src/frontend/app.py

El mapa de celdas y el buscador con sugerencias son componentes JavaScript propios (src/frontend/components).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ALTO_RECALL, CELL_SIZE_M, RISK_LEVELS, SEARCH_POSTAL_CODES, SNCZI_COLUMN, TARGET_RECALL  # noqa: E402
from src.backend import services  # noqa: E402
from src.frontend import components  # noqa: E402

# Paleta apta para daltonismo (azul-naranja-rojo); el nivel también va siempre en texto
COLORS = {"Bajo": "#2C7BB6", "Medio": "#E66101", "Alto": "#B2182B"}
# Bajo visible pero suave, para que el mapa resalte el riesgo sin que parezca que faltan celdas
OPACITY = {"Bajo": 0.35, "Medio": 0.55, "Alto": 0.7}
OBSERVED_COLORS = {"Grave": "#08306B", "Moderada": "#2171B5", "Leve": "#6BAED6"}
DISCLAIMER = (
    "**Herramienta informativa que no sustituye los avisos de las autoridades.** "
    f"Muestra la susceptibilidad estimada de cada celda de {CELL_SIZE_M} m ante un episodio tipo DANA, "
    "no una predicción del nivel del agua. Ante una emergencia, siga las indicaciones de Protección Civil, el 112 y ES-Alert."
)

# Vista inicial: la zona cero de la DANA (Paiporta, Picanya, Sedaví, Benetússer, Catarroja...)
DEFAULT_CENTER, DEFAULT_ZOOM = (39.415, -0.410), 12
# Zoom al buscar (escala de MapLibre): un municipio, distrito o barrio se ve entero a 14; una dirección, con sus
# calles, a 15
PLACE_ZOOM, ADDRESS_ZOOM = 14, 15
MAP_HEIGHT_PX = 560
M_PER_KM = 1000

st.set_page_config(page_title="DANA Predictor", page_icon="🌧️", layout="wide")


@st.cache_resource
def risk_map() -> pd.DataFrame:
    """Celdas con su predicción y lo observado en la DANA, sin geometría (el mapa la construye en el navegador).

    Se comparte entre sesiones sin copiarla en cada ejecución: no se modifica."""
    return pd.DataFrame(services.get_risk_map().drop(columns="geometry"))


@st.cache_data
def model_card() -> dict:
    return services.get_model_card()


@st.cache_resource
def map_grid() -> dict:
    return services.get_map_grid()


@st.cache_resource
def places() -> list[dict]:
    """Municipios, distritos y barrios con su centro, para las sugerencias del buscador."""
    return services.list_places().to_dict("records")


def legend() -> None:
    """Leyenda de colores de los niveles de riesgo y explicación de las capas del mapa."""
    chips = " ".join(
        f"<span style='display:inline-block;margin-right:12px'>"
        f"<span style='display:inline-block;width:14px;height:14px;background:{COLORS[level]};"
        f"border:1px solid #333;vertical-align:middle'></span> {level}</span>"
        for level in RISK_LEVELS
    )
    st.markdown(f"**Leyenda (nivel de riesgo estimado):** {chips}", unsafe_allow_html=True)
    st.caption(
        f"Cada color es una celda de {CELL_SIZE_M} m. Pulsa una celda para ver su ficha. Con el control de capas (arriba "
        "a la derecha) se puede ocultar cada capa o mostrar lo inundado en la DANA según Copernicus EMS y la huella de "
        "la GVA (contorno azul discontinuo; oscuro = más grave). Los arrozales y el agua permanente (el lago de la "
        "Albufera, los cauces de los ríos) no se incluyen: no son suelo urbano."
    )


def zone_detail(zone_id: int, data, card: dict) -> None:
    """Ficha de la celda elegida: nivel, probabilidad, variables, mapas oficiales y lo observado en la DANA."""
    row = data[data["zone_id"] == zone_id]
    if row.empty:
        st.info("La ubicación está fuera del área piloto (València, L'Horta Sud y L'Horta Oest).")
        return
    row = row.iloc[0]
    st.subheader(f"{row['name']} · {row['place']}")
    if pd.notna(row["district"]):
        st.caption(f"Distrito {row['district']} · {row['municipality']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Susceptibilidad estimada", str(row["risk_level"]))
    c2.metric("Posición en el área piloto", f"más que el {row['risk_percentile']} %",
              help="Porcentaje de celdas del piloto con menor susceptibilidad que esta.")
    c3.metric("Observado en la DANA", row["observed"])
    if row["risk_level"] == "Bajo":
        st.info(
            "**Bajo no significa sin riesgo.** El modelo solo conoce el desbordamiento de barrancos de la DANA de 2024: "
            "no sabe nada de la lluvia local intensa ni de la capacidad del alcantarillado."
        )
    flooded_share = card["flooded_cells"] / card["cells"]
    st.markdown(
        f"El nivel es **relativo** al área piloto y está aprendido de la DANA de 2024: *Medio* agrupa las celdas que "
        f"el modelo necesita marcar para encontrar el {TARGET_RECALL:.0%} de lo inundado en la DANA y *Alto*, las de "
        f"mayor probabilidad, que concentran la mitad ({ALTO_RECALL:.0%}) de lo inundado. La probabilidad calibrada de "
        f"esta celda es {row['risk_probability']:.0%} (en el área piloto se inundó el {flooded_share:.0%} de las celdas)."
    )
    patricova = f"nivel {row['patricova_level']}" if row["patricova_level"] else "no"
    snczi = "sí" if row[SNCZI_COLUMN] else "no"
    st.caption(
        f"Cauce principal más cercano: {row['channel_name']} a {row['dist_channel_km'] * M_PER_KM:.0f} m · "
        f"altitud {row['altitude_avg']:.1f} m · pendiente {row['slope_avg']:.2f} % · "
        f"impermeable {row['imperviousness']:.0f} % · geología: {row['geology']} · "
        f"en mapas oficiales: PATRICOVA {patricova}, SNCZI T=500 {snczi}"
    )
    history = services.get_zone_history(zone_id)
    if not history.empty:
        st.markdown("**Inundación observada en la celda**")
        st.dataframe(history, hide_index=True, width="stretch")


def component_event(key: str, event: str):
    """Último valor que un componente JavaScript ha enviado para ese evento (o None)."""
    state = st.session_state.get(key)
    return state.get(event) if state is not None else None


def on_place() -> None:
    """Lugar elegido en el buscador: centra el mapa en él y selecciona su celda, antes de volver a dibujar la página."""
    place = component_event("buscador", "place")
    if not place:
        return
    is_own_place = place["kind"] == "place"
    zone = services.find_zone(place["lat"], place["lon"])
    st.session_state.update(
        center=(place["lat"], place["lon"]), zoom=PLACE_ZOOM if is_own_place else ADDRESS_ZOOM,
        search_label=f"{place['detail']} {place['label']}" if is_own_place else place["label"], zone=zone,
        # El mapa solo se mueve cuando cambia view_id: al buscar, no al pulsar una celda
        view_id=st.session_state.get("view_id", 0) + 1,
        # Un municipio o barrio puede tener el centro en un hueco sin celdas (un arrozal): solo se avisa de las direcciones
        search_outside=zone is None and not is_own_place,
    )


def search_box() -> None:
    """Buscador de la barra lateral con sugerencias mientras se escribe: municipios, distritos y barrios propios, y
    calles y direcciones de CartoCiudad. Al elegir una, el mapa se centra en ella."""
    with st.sidebar:
        components.place_search(
            places=places(), postal_codes=SEARCH_POSTAL_CODES, label="Buscar dirección, calle, barrio o municipio",
            placeholder="Ej.: Plaza del Ayuntamiento", key="buscador", on_place=on_place,
        )
        if st.session_state.get("search_outside"):
            st.warning("No se ha encontrado esa ubicación dentro del área piloto.")


def main() -> None:
    """Página completa: aviso, buscador, recuento de celdas por nivel y las pestañas Mapa, Tabla de zonas y Modelo y
    datos."""
    st.title("DANA Predictor")
    st.caption(
        "Susceptibilidad a inundación en València, L'Horta Sud y L'Horta Oest, aprendida de la DANA de 2024 · prototipo académico"
    )
    st.warning(DISCLAIMER, icon="⚠️")

    card = model_card()
    st.sidebar.header("Consulta")
    search_box()

    data = risk_map()
    counts = data["risk_level"].value_counts()
    cols = st.columns(4)
    for col, level in zip(cols[: len(RISK_LEVELS)], RISK_LEVELS, strict=True):
        col.metric(f"Celdas en nivel {level}", int(counts.get(level, 0)))
    cols[3].metric("Modelo", card["model_version"])

    tab_map, tab_table, tab_model = st.tabs(["Mapa", "Tabla de zonas", "Modelo y datos"])

    with tab_map:
        legend()
        if "search_label" in st.session_state:
            st.caption(f"Ubicación buscada: {st.session_state['search_label']}")
        map_and_detail(data, card)

    with tab_table:
        zones_table()

    with tab_model:
        model_tab(card)


def on_cell() -> None:
    """Celda pulsada en el mapa (0 si fuera de las celdas), antes de volver a ejecutar el fragmento: así la ficha y la
    celda marcada coinciden en la misma pasada."""
    zone = component_event("mapa", "cell")
    st.session_state.zone = int(zone) if zone else None


@st.fragment
def map_and_detail(data, card: dict) -> None:
    """Mapa y ficha de la celda. Es un fragmento: al pulsar una celda solo se vuelve a ejecutar esta parte, no la
    página entera (tabla de todas las celdas, ficha del modelo...), y la respuesta es mucho más rápida."""
    components.cell_map(
        grid=map_grid(), colors=COLORS, opacity=OPACITY, observed_colors=OBSERVED_COLORS,
        selected=st.session_state.get("zone"),
        view={"id": st.session_state.get("view_id", 0), "center": list(st.session_state.get("center", DEFAULT_CENTER)),
              "zoom": st.session_state.get("zoom", DEFAULT_ZOOM)},
        height=MAP_HEIGHT_PX, key="mapa", on_cell=on_cell,
    )
    if st.session_state.get("zone"):
        zone_detail(st.session_state["zone"], data, card)
    else:
        st.caption("Pulse una celda del mapa o use el buscador para ver su detalle.")


# Columnas de la tabla accesible y su nombre para el usuario
TABLE_COLUMNS = {
    "name": "Celda", "municipality": "Municipio", "place": "Barrio / municipio", "risk_level": "Nivel",
    "risk_probability": "Probabilidad", "risk_percentile": "Percentil", "observed": "Observado en la DANA",
    "dist_channel_km": "Distancia al cauce (km)", "altitude_avg": "Altitud (m)", "imperviousness": "Impermeable (%)",
    "geology": "Geología", "patricova_level": "PATRICOVA",
}


@st.cache_resource
def table_and_csv() -> tuple[pd.DataFrame, bytes]:
    """Tabla de todas las celdas, ordenada, y su CSV de descarga; se preparan una sola vez.

    Los textos que se repiten (municipio, barrio, nivel, geología...) van como categorías: la tabla que se envía al
    navegador pesa varias veces menos y se ve igual."""
    table = risk_map()[list(TABLE_COLUMNS)].sort_values(["risk_level", "risk_probability"], ascending=False)
    table = table.rename(columns=TABLE_COLUMNS).reset_index(drop=True)
    for column in table.select_dtypes(include=["object", "string"]).columns:
        table[column] = table[column].astype("category")
    return table, table.to_csv(index=False).encode()


def zones_table() -> None:
    """Alternativa accesible al mapa: todas las celdas, ordenables, y el resumen por distrito o municipio."""
    st.markdown("Alternativa accesible al mapa: zonas ordenables por nivel y probabilidad.")
    table, csv = table_and_csv()
    st.dataframe(table, hide_index=True, width="stretch")
    st.markdown("**Resumen por distrito** (municipio fuera de València)")
    st.dataframe(services.get_district_summary(), hide_index=True, width="stretch")
    st.download_button("Descargar CSV", csv, file_name="susceptibilidad_celdas.csv")


def model_tab(card: dict) -> None:
    """Ficha del modelo: métricas, comparación con los mapas oficiales, zonas no vistas, calados y detalles."""
    model_metrics(card)
    official_maps_comparison(card)
    unseen_zones(card)
    field_depths(card)
    model_details(card)


def model_metrics(card: dict) -> None:
    """Versión, qué predice, cómo se valida y métricas del modelo elegido frente a los objetivos."""
    st.markdown(f"**Versión:** {card['model_version']} · **modelo:** {card['chosen_model']} · entrenado {card['trained_at']}")
    st.markdown(f"**Qué predice:** {card['target']}. **Validación:** {card['validation_method']}.")
    st.markdown(f"**Selección:** {card['selection_rule']}.")
    criteria = card["success_criteria"]
    val = card["models"][card["chosen_model"]]["validation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (validación)", f"{val['recall']:.2f}", help=f"Objetivo ≥ {criteria['recall_target']}")
    c2.metric("F1 (validación)", f"{val['f1']:.2f}", help=f"Objetivo ≥ {criteria['f1_target']}")
    c3.metric("ROC-AUC", f"{val['roc_auc']:.3f}")
    c4.metric("PR-AUC", f"{val['pr_auc']:.3f}")
    st.caption(
        f"Referencia trivial: marcar todas las celdas como inundadas daría F1 {card['mark_all_f1']:.2f}, porque se inundó "
        f"el {card['flooded_cells'] / card['cells']:.0%} del área. El F1 hay que leerlo frente a esa cifra; el ROC-AUC "
        "no depende de cuántas celdas se inundaron."
    )


def official_maps_comparison(card: dict) -> None:
    """Tabla con los modelos, los mapas oficiales tal cual y el modelo con PATRICOVA como variable."""
    keys = ("recall", "f1", "precision", "roc_auc", "pr_auc")

    def row(label: str, result: dict) -> dict:
        return {"modelo": label, **{k: result["validation"][k] for k in keys}}

    rows = [row(name, r) for name, r in card["models"].items()]
    rows += [row(f"mapa oficial tal cual: {name}", r) for name, r in card["baselines"].items()]
    rows.append(row(f"{card['chosen_model']} con PATRICOVA como variable", card["ablation_with_patricova"]))
    st.markdown("**Modelos frente a los mapas oficiales** (validación cruzada por bloques)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "El modelo no usa PATRICOVA ni el SNCZI: aprende solo del terreno, el suelo, los cauces y la geología, y se "
        "compara con ellos. La fila \"con PATRICOVA\" muestra cuánto añadiría usarla."
    )


def unseen_zones(card: dict) -> None:
    """Resultado de la prueba externa en la Ribera, una cuenca que el modelo no ha visto."""
    def test_row(name: str, r: dict) -> dict:
        refs = r["references"]
        return {"zona sin ver": name, "celdas": r["test_cells"], "% inundado": r["test_flooded_share"],
                "F1 modelo": r["model"]["f1"], "ROC-AUC modelo": r["model"]["roc_auc"],
                "F1 PATRICOVA": refs["PATRICOVA"]["f1"], "F1 SNCZI T=500": refs["SNCZI T=500"]["f1"],
                "ROC-AUC SNCZI T=500": refs["SNCZI T=500"]["roc_auc"], "F1 marcar todo": r["mark_all_f1"]}

    ext = card["external_test"]
    st.markdown("**Prueba externa: una zona que el modelo no ha visto**")
    st.dataframe(pd.DataFrame([test_row(f"{ext['region']} (otra cuenca: Magro y Xúquer)", ext)]), hide_index=True,
                 width="stretch")
    st.caption(
        f"La Ribera nunca entra en el entrenamiento ni en el mapa: el modelo final la predice solo para medir si "
        f"generaliza (celdas a más de {ext['gap_m']} m del área piloto). La validación por bloques de 1 km mide otra "
        "cosa: si el modelo rellena bien los huecos dentro de una zona que conoce, que es lo que hace el mapa."
    )


def field_depths(card: dict) -> None:
    """Puntos con calado medido en campo que caen en celdas Medio o Alto, frente a los mapas oficiales."""
    field = card["field_check"]
    st.markdown("**Comprobación con calados medidos en campo** (GVA: IVE, SGRT, UPV)")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric(f"Puntos con ≥ {field['min_depth_m']} m de agua", field["points"], help=f"En {field['cells_with_points']} celdas")
    f2.metric("En celdas Medio o Alto del modelo", f"{field['model_detected_pct']:.0f} %")
    f3.metric("Dentro de PATRICOVA", f"{field['references_detected_pct']['PATRICOVA']:.0f} %")
    f4.metric("Dentro del SNCZI T=500", f"{field['references_detected_pct']['SNCZI T=500']:.0f} %")
    st.caption(
        "No es una validación independiente: la huella de la GVA, que forma parte de la etiqueta, cubre casi todos "
        "estos puntos. Sirve para ver cuánto de lo que se midió en campo quedaba dentro de los mapas oficiales."
    )


def model_details(card: dict) -> None:
    """ROC-AUC por grupo, falsos negativos por distrito, umbrales, importancia de las variables, fuentes y
    limitaciones."""
    st.markdown("**ROC-AUC por grupo de bloques** (cuánto varía según la zona)")
    st.dataframe(pd.DataFrame({name: r["roc_auc_by_fold"] for name, r in card["models"].items()}), width="stretch")

    st.markdown("**Falsos negativos por distrito o municipio** (muestra pequeña, interpretar con cautela)")
    st.dataframe(pd.DataFrame(card["false_negatives_by_district"]), hide_index=True, width="stretch")

    st.markdown(
        f"**Umbrales (fijados en validación):** Medio ≥ {card['thresholds']['medio']:.2f} (Medio + Alto encuentran el "
        f"{TARGET_RECALL:.0%} de lo inundado) · Alto ≥ {card['thresholds']['alto']:.2f} (concentra el {ALTO_RECALL:.0%} "
        "de lo inundado, la parte de mayor probabilidad)"
    )
    st.markdown(f"**Variables utilizadas** (importancia por {card['feature_importance_method']})")
    st.dataframe(pd.DataFrame(card["feature_importance"].items(), columns=["variable", "importancia"]), hide_index=True)

    st.markdown("**Fuentes de datos** (todas reales)")
    st.dataframe(services.get_data_sources().drop(columns=["key"]), hide_index=True, width="stretch")
    st.markdown(
        "**Limitaciones:** el modelo aprende de un único episodio (la DANA del 29-10-2024): es un mapa de "
        "susceptibilidad aprendido de esa DANA, no una predicción del riesgo con cualquier lluvia. En otra cuenca que no "
        "ha visto (la Ribera) no supera a los mapas oficiales: la geología tendría sentido allí, pero en València y "
        "L'Horta casi no hay llanura de inundación de la que aprenderlo. La comparación con PATRICOVA y el SNCZI dentro "
        "del área favorece algo al modelo, que aprende de esta DANA; los mapas oficiales son anteriores. No usa la "
        "lluvia. La etiqueta une el satélite (Copernicus EMS), que no ve el agua entre edificios, y la huella oficial "
        "de la GVA, que cubre los cascos urbanos. Los arrozales y el agua permanente (lago de la Albufera, cauces; "
        "Corine Land Cover) se excluyen: no son suelo urbano y el satélite no distingue esa agua de la inundación. La "
        "validación por bloques de 1 km es algo optimista y el F1 sube por la proporción de celdas inundadas (ver "
        "referencia trivial). La distancia al cauce se diseñó revisando este mismo episodio. Sin datos de drenaje, el "
        "modelo no representa la red de alcantarillado."
    )


main()
