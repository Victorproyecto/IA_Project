"""Frontend DANA Predictor (Streamlit). Uso: streamlit run src/frontend/app.py

El mapa de celdas, el buscador con sugerencias y el botón de tema son componentes JavaScript propios
(src/frontend/components).
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    ALTO_RECALL, CELL_SIZE_M, GEOLOGY_LABELS, RISK_LEVELS, SEARCH_POSTAL_CODES, SNCZI_COLUMN, TARGET_RECALL,
)
from src.backend import services  # noqa: E402
from src.frontend import components  # noqa: E402

# Paleta apta para daltonismo (azul-naranja-rojo); el nivel también va siempre en texto
COLORS = {"Bajo": "#2C7BB6", "Medio": "#E66101", "Alto": "#B2182B"}
# Bajo visible pero suave, para que el mapa resalte el riesgo sin que parezca que faltan celdas
OPACITY = {"Bajo": 0.35, "Medio": 0.55, "Alto": 0.7}
# Lo inundado en la DANA, en morados (no se confunde con el azul, el naranja y el rojo de la susceptibilidad): más
# intenso = más grave. En tema oscuro, lo más grave es lo más claro, porque un morado oscuro no se ve sobre el fondo
OBSERVED_COLORS = {"Leve": "#BCBDDC", "Moderada": "#807DBA", "Grave": "#4A1486"}
OBSERVED_COLORS_DARK = {"Leve": "#5E4B8B", "Moderada": "#8F7BC8", "Grave": "#B9A3F2"}
OBSERVED_OPACITY = {"Leve": 0.6, "Moderada": 0.7, "Grave": 0.8}
DISCLAIMER = (
    "**Herramienta informativa: no sustituye los avisos de las autoridades.** Ante una emergencia, siga las "
    "indicaciones de Protección Civil, el 112 y ES-Alert."
)

# Vista inicial: la zona cero de la DANA (Paiporta, Picanya, Sedaví, Benetússer, Catarroja...)
DEFAULT_CENTER, DEFAULT_ZOOM = (39.415, -0.410), 12
# Zoom al buscar (escala de MapLibre): un municipio, distrito o barrio se ve entero a 14; una dirección, con sus
# calles, a 15
PLACE_ZOOM, ADDRESS_ZOOM = 14, 15
MAP_HEIGHT_PX = 600
# Mapa a la izquierda y leyenda y ficha de la celda a la derecha (en el móvil, una debajo de otra)
MAP_COLUMNS = [3, 2]
M_PER_KM = 1000
# Nombres para el usuario de las variables y los modelos (pestaña del modelo)
FEATURE_LABELS = {
    "altitude_avg": "Altitud media",
    "slope_avg": "Pendiente",
    "imperviousness": "Suelo sellado",
    "dist_channel_km": "Distancia al cauce principal",
    "height_above_channel_m": "Altura sobre el cauce",
    **{group: f"Geología: {label}" for group, label in GEOLOGY_LABELS.items()},
}
MODEL_LABELS = {"logistic_regression": "Regresión logística", "random_forest": "Random Forest"}
METRIC_LABELS = {"recall": "Recall", "f1": "F1", "precision": "Precisión", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}

st.set_page_config(page_title="DANA Predictor", page_icon="🌧️", layout="wide")


# --- Formato de los números en español (coma decimal y punto de miles) ------------------------------------------

def es(value: float, decimals: int = 0) -> str:
    """Número como se escribe en español: 29.471 o 0,72."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def pct(share: float, decimals: int = 0) -> str:
    """Fracción (0-1) como porcentaje en español: 0,236 -> «24 %»."""
    return f"{es(100 * share, decimals)} %"


# --- Datos (se cargan una vez y se comparten entre sesiones) ----------------------------------------------------

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
def area_outline() -> dict:
    return services.get_area_outline()


@st.cache_resource
def places() -> list[dict]:
    """Municipios, distritos y barrios con su centro, para las sugerencias del buscador."""
    return services.list_places().to_dict("records")


def dark_theme() -> bool:
    return st.context.theme.type == "dark"


def observed_colors() -> dict:
    """Colores de la gravedad de lo inundado según el tema."""
    return OBSERVED_COLORS_DARK if dark_theme() else OBSERVED_COLORS


# --- Página -----------------------------------------------------------------------------------------------------

def main() -> None:
    """Página completa: cabecera con el aviso y las pestañas Mapa, Tabla de zonas y Modelo y datos."""
    # Menos margen encima del título que el de Streamlit (6 rem), para que el mapa quede a la vista al entrar
    st.markdown("<style>[data-testid='stMainBlockContainer']{padding-top:2.5rem}</style>", unsafe_allow_html=True)
    title_col, theme_col = st.columns([5, 1], vertical_alignment="center")
    title_col.title("DANA Predictor")
    with theme_col:
        components.theme_toggle(key="tema")
    st.caption(
        "Susceptibilidad a inundación en València, L'Horta Sud y L'Horta Oest, aprendida de la DANA de 2024 · prototipo académico"
    )
    st.warning(DISCLAIMER, icon="⚠️")

    card = model_card()
    data = risk_map()
    tab_map, tab_table, tab_model = st.tabs(["Mapa", "Tabla de zonas", "Modelo y datos"])
    with tab_map:
        search_box()
        map_and_detail(data, card)
        how_to_read()
    with tab_table:
        zones_table()
    with tab_model:
        model_tab(card)


# --- Pestaña Mapa: buscador, mapa, leyenda y ficha de la celda --------------------------------------------------

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
    """Buscador encima del mapa (también visible en el móvil) con sugerencias mientras se escribe: municipios,
    distritos y barrios propios, y calles y direcciones de CartoCiudad. Al elegir una, el mapa se centra en ella."""
    search_col, _ = st.columns(MAP_COLUMNS)
    with search_col:
        components.place_search(
            places=places(), postal_codes=SEARCH_POSTAL_CODES, label="Buscar dirección, calle, barrio o municipio",
            placeholder="Ej.: Plaza del Ayuntamiento", key="buscador", on_place=on_place,
        )
        if st.session_state.get("search_outside"):
            st.warning("No se ha encontrado esa ubicación dentro del área piloto.")


def on_cell() -> None:
    """Celda pulsada en el mapa (0 si fuera de las celdas), antes de volver a ejecutar el fragmento: así la ficha y la
    celda marcada coinciden en la misma pasada."""
    zone = component_event("mapa", "cell")
    st.session_state.zone = int(zone) if zone else None


@st.fragment
def map_and_detail(data, card: dict) -> None:
    """Mapa a la izquierda; leyenda y ficha de la celda a la derecha, a la vista al pulsar. Es un fragmento: al pulsar
    una celda solo se vuelve a ejecutar esta parte, no la página entera, y la respuesta es mucho más rápida."""
    map_col, detail_col = st.columns(MAP_COLUMNS, gap="medium")
    with map_col:
        components.cell_map(
            grid=map_grid(), colors=COLORS, opacity=OPACITY, observed_colors=observed_colors(),
            observed_opacity=OBSERVED_OPACITY, area=area_outline(),
            dark=dark_theme(), selected=st.session_state.get("zone"),
            view={"id": st.session_state.get("view_id", 0),
                  "center": list(st.session_state.get("center", DEFAULT_CENTER)),
                  "zoom": st.session_state.get("zoom", DEFAULT_ZOOM)},
            height=MAP_HEIGHT_PX, key="mapa", on_cell=on_cell,
        )
    with detail_col:
        legend(data)
        if st.session_state.get("zone"):
            zone_detail(st.session_state["zone"], data, card)
        else:
            st.info("Pulsa una celda del mapa o busca una dirección para ver su ficha.")


def legend(data: pd.DataFrame) -> None:
    """Leyenda compacta: color y número de celdas de cada nivel y colores de lo inundado en la DANA."""
    counts = data["risk_level"].value_counts()
    box = "display:inline-block;width:14px;height:14px;vertical-align:middle;margin-right:4px"
    levels = " ".join(
        f"<span style='white-space:nowrap;margin-right:10px'><span style='{box};background:{COLORS[level]};"
        f"border:1px solid #333'></span>{level} ({es(counts.get(level, 0))})</span>"
        for level in RISK_LEVELS
    )
    not_evaluated = (f"<span style='white-space:nowrap'><span style='{box};border:1px dashed currentColor'></span>"
                     "Sin color: no evaluado</span>")
    severities = " ".join(
        f"<span style='white-space:nowrap;margin-right:10px'><span style='{box};background:{color};"
        f"border:1px solid #333'></span>{severity}</span>"
        for severity, color in observed_colors().items()
    )
    st.markdown(f"**Nivel de riesgo estimado (celdas):** {levels} {not_evaluated}", unsafe_allow_html=True)
    st.markdown(f"**Inundado en la DANA** (actívalo en el control de capas): {severities}", unsafe_allow_html=True)
    if "search_label" in st.session_state:
        st.caption(f"Ubicación buscada: {st.session_state['search_label']}")


def how_to_read() -> None:
    """Explicación del mapa, plegada para que el mapa quede a la vista al entrar."""
    with st.expander("Cómo leer el mapa"):
        st.caption(
            f"Cada color es una celda de {CELL_SIZE_M} m: la susceptibilidad estimada ante un episodio tipo DANA, no una "
            "predicción del nivel del agua. Pulsa una celda para ver su ficha. Con el control de capas (arriba a la "
            "derecha) se puede ocultar cada capa o mostrar el límite de València, L'Horta Sud y L'Horta Oest y lo "
            "inundado en la DANA según Copernicus EMS y la huella de la GVA, en morado (más intenso = más grave; para "
            "compararlo con la susceptibilidad, alterna las dos capas). Las zonas sin color no se evalúan: el mar, los arrozales y el agua "
            "permanente (el lago de la Albufera, los cauces de los ríos). El mapa se centra en el suelo urbano y, en los "
            "arrozales, el satélite no distingue el agua del cultivo de la de la inundación."
        )


def zone_detail(zone_id: int, data, card: dict) -> None:
    """Ficha de la celda elegida: nivel, posición, terreno, mapas oficiales y lo observado en la DANA."""
    row = data[data["zone_id"] == zone_id]
    if row.empty:
        st.info("La ubicación está fuera del área piloto (València, L'Horta Sud y L'Horta Oest).")
        return
    row = row.iloc[0]
    st.subheader(row["place"], divider="gray")
    where = [f"distrito {row['district']}" if pd.notna(row["district"]) else None,
             row["municipality"] if row["municipality"] != row["place"] else None, row["name"].lower()]
    st.caption(" · ".join(part for part in where if part))
    level_col, observed_col = st.columns(2)
    level_col.metric("Susceptibilidad estimada", str(row["risk_level"]))
    observed_col.metric("Observado en la DANA", row["observed"])
    st.markdown(
        f"Más susceptible que el **{row['risk_percentile']} %** de las celdas del área · probabilidad calibrada "
        f"**{pct(row['risk_probability'])}**"
    )
    if row["risk_level"] == "Bajo":
        st.info(
            "**Bajo no significa sin riesgo.** El modelo solo conoce el desbordamiento de barrancos de la DANA de 2024: "
            "no sabe nada de la lluvia local intensa ni de la capacidad del alcantarillado."
        )
    with st.expander("¿Qué significa el nivel?"):
        st.markdown(
            f"El nivel es **relativo** al área piloto y está aprendido de la DANA de 2024: *Medio* agrupa las celdas que "
            f"el modelo necesita marcar para encontrar el {pct(TARGET_RECALL)} de lo inundado en la DANA y *Alto*, las de "
            f"mayor probabilidad, que concentran la mitad ({pct(ALTO_RECALL)}) de lo inundado. En el área piloto se "
            f"inundó el {pct(card['flooded_cells'] / card['cells'])} de las celdas."
        )
    terrain_and_official_maps(row)
    observed_in_cell(zone_id)


def terrain_and_official_maps(row: pd.Series) -> None:
    """Variables del terreno de la celda, con sus unidades, y si está dentro de los mapas oficiales."""
    patricova = f"nivel {row['patricova_level']} (el 1 es el más peligroso de 6)" if row["patricova_level"] else "fuera"
    terrain = {
        "Altitud media": f"{es(row['altitude_avg'], 1)} m",
        "Altura sobre el cauce al que drena": f"{es(row['height_above_channel_m'], 1)} m",
        "Distancia al cauce principal": f"{es(row['dist_channel_km'] * M_PER_KM)} m ({row['channel_name']})",
        "Pendiente": f"{es(row['slope_avg'], 1)} %",
        "Suelo sellado": f"{es(row['imperviousness'])} %",
        "Geología": row["geology"],
    }
    rows = "".join(f"<tr><td style='padding:4px 12px 4px 0'>{name}</td><td style='padding:4px 0'><b>{value}</b></td></tr>"
                   for name, value in terrain.items())
    st.markdown("**Terreno de la celda** (lo que usa el modelo)")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'>{rows}</table>", unsafe_allow_html=True)
    st.markdown(
        "**Mapas oficiales de peligrosidad** (solo referencia: el modelo no los usa)\n\n"
        f"- PATRICOVA: {patricova}\n"
        f"- SNCZI, zona inundable de 500 años: {'dentro' if row[SNCZI_COLUMN] else 'fuera'}"
    )
    st.caption(
        "La altura sobre el cauce es la altura de la celda sobre el cauce, el mar o la Albufera al que llega su agua "
        "siguiendo la pendiente: cuanto más baja, más fácil que el agua desbordada llegue a ella."
    )


def observed_in_cell(zone_id: int) -> None:
    """Lo que se inundó en la celda durante la DANA: gravedad, parte de la celda y fuente."""
    history = services.get_zone_history(zone_id)
    for event in history.itertuples():
        st.markdown(
            f"**Inundación observada ({event.episode}):** {event.severity.lower()}, en el "
            f"{pct(event.flooded_fraction)} de la celda · fuente: {event.source}"
        )


# --- Pestaña Tabla de zonas ------------------------------------------------------------------------------------

# Columnas de la tabla accesible y su nombre para el usuario
TABLE_COLUMNS = {
    "municipality": "Municipio", "place": "Barrio / municipio", "risk_level": "Nivel",
    "probability_pct": "Probabilidad (%)", "risk_percentile": "Percentil", "observed": "Observado en la DANA",
    "height_above_channel_m": "Altura sobre el cauce (m)", "dist_channel_m": "Distancia al cauce (m)",
    "altitude_avg": "Altitud (m)", "imperviousness": "Suelo sellado (%)", "geology": "Geología",
    "patricova_level": "PATRICOVA", "name": "Código de celda",
}
# Los números se muestran con el formato del navegador (coma decimal en español) y siguen siendo ordenables
LOCALIZED = st.column_config.NumberColumn(format="localized")
TABLE_FORMATS = {name: LOCALIZED for name in ("Altura sobre el cauce (m)", "Altitud (m)", "Distancia al cauce (m)")}
DISTRICT_COLUMNS = {"zona": "Distrito o municipio", "Bajo": "Celdas en Bajo", "Medio": "Celdas en Medio",
                    "Alto": "Celdas en Alto", "probabilidad_media": "Probabilidad media (%)",
                    "inundadas_dana": "Inundadas en la DANA"}


@st.cache_resource
def table_and_csv() -> tuple[pd.DataFrame, bytes]:
    """Tabla de todas las celdas, ordenada, y su CSV de descarga (con todos los decimales); se preparan una sola vez.

    Los textos que se repiten (municipio, barrio, nivel, geología...) van como categorías: la tabla que se envía al
    navegador pesa varias veces menos y se ve igual."""
    cells = risk_map().sort_values(["risk_level", "risk_probability"], ascending=False)
    exact = cells.assign(probability_pct=100 * cells["risk_probability"], dist_channel_m=cells["dist_channel_km"] * M_PER_KM)
    csv = exact[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS).to_csv(index=False).encode()
    table = cells.assign(
        probability_pct=(100 * cells["risk_probability"]).round().astype(int),
        dist_channel_m=(cells["dist_channel_km"] * M_PER_KM).round().astype(int),
        altitude_avg=cells["altitude_avg"].round(1), height_above_channel_m=cells["height_above_channel_m"].round(1),
        imperviousness=cells["imperviousness"].round().astype(int),
    )[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS).reset_index(drop=True)
    for column in table.select_dtypes(include=["object", "string"]).columns:
        table[column] = table[column].astype("category")
    return table, csv


def zones_table() -> None:
    """Alternativa accesible al mapa: todas las celdas, ordenables, y el resumen por distrito o municipio."""
    st.markdown("Alternativa accesible al mapa: todas las celdas, ordenables por cualquier columna.")
    table, csv = table_and_csv()
    st.dataframe(table, hide_index=True, width="stretch", column_config=TABLE_FORMATS)
    st.download_button("Descargar CSV", csv, file_name="susceptibilidad_celdas.csv")
    st.markdown("**Resumen por distrito** (municipio fuera de València)")
    summary = services.get_district_summary()
    summary = summary.assign(probabilidad_media=(100 * summary["probabilidad_media"]).round().astype(int))
    st.dataframe(summary.rename(columns=DISTRICT_COLUMNS), hide_index=True, width="stretch")


# --- Pestaña Modelo y datos ------------------------------------------------------------------------------------

def model_tab(card: dict) -> None:
    """Ficha del modelo: métricas, comparación con los mapas oficiales, zonas no vistas, calados y detalles."""
    model_metrics(card)
    official_maps_comparison(card)
    unseen_zones(card)
    field_depths(card)
    model_details(card)


def model_metrics(card: dict) -> None:
    """Versión, qué predice, cómo se valida y métricas del modelo elegido frente a los objetivos."""
    chosen = MODEL_LABELS.get(card["chosen_model"], card["chosen_model"])
    st.markdown(f"**Versión:** {card['model_version']} · **modelo:** {chosen} · entrenado {card['trained_at']}")
    st.markdown(f"**Qué predice:** {card['target']}. **Validación:** {card['validation_method']}.")
    st.markdown(f"**Selección:** {card['selection_rule']}.")
    criteria = card["success_criteria"]
    val = card["models"][card["chosen_model"]]["validation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (validación)", es(val["recall"], 2),
              **target_delta(criteria["validation_recall_ok"], criteria["recall_target"]))
    c2.metric("F1 (validación)", es(val["f1"], 2), **target_delta(criteria["validation_f1_ok"], criteria["f1_target"]))
    c3.metric("ROC-AUC", es(val["roc_auc"], 3))
    c4.metric("PR-AUC", es(val["pr_auc"], 3))
    st.caption(
        f"Referencia trivial: marcar todas las celdas como inundadas daría F1 {es(card['mark_all_f1'], 2)}, porque se "
        f"inundó el {pct(card['flooded_cells'] / card['cells'])} del área. El F1 hay que leerlo frente a esa cifra; el "
        "ROC-AUC no depende de cuántas celdas se inundaron."
    )


def target_delta(met: bool, target: float) -> dict:
    """Argumentos de st.metric que indican, sin flecha, si se cumple el objetivo de la Fase 1 (verde) o no (rojo)."""
    return {"delta": f"{'cumple' if met else 'no cumple'} el objetivo (≥ {es(target, 2)})",
            "delta_color": "green" if met else "red", "delta_arrow": "off"}


def official_maps_comparison(card: dict) -> None:
    """Tabla con los modelos, los mapas oficiales tal cual y el modelo con PATRICOVA como variable."""
    def row(label: str, result: dict) -> dict:
        return {"Modelo": label, **{name: es(result["validation"][key], 3) for key, name in METRIC_LABELS.items()}}

    rows = [row(MODEL_LABELS.get(name, name), r) for name, r in card["models"].items()]
    rows += [row(f"Mapa oficial tal cual: {name}", r) for name, r in card["baselines"].items()]
    chosen = MODEL_LABELS.get(card["chosen_model"], card["chosen_model"])
    rows.append(row(f"{chosen} con PATRICOVA como variable", card["ablation_with_patricova"]))
    st.markdown("**Modelos frente a los mapas oficiales** (validación cruzada por bloques)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "El modelo no usa PATRICOVA ni el SNCZI: aprende solo del terreno, el suelo, los cauces y la geología, y se "
        "compara con ellos. La fila \"con PATRICOVA\" muestra cuánto añadiría usarla."
    )


def unseen_zones(card: dict) -> None:
    """Resultado de la prueba externa en la Ribera, una cuenca que el modelo no ha visto."""
    ext = card["external_test"]
    refs = ext["references"]
    row = {"Zona sin ver": f"{ext['region']} (otra cuenca: Magro y Xúquer)", "Celdas": es(ext["test_cells"]),
           "Inundadas": pct(ext["test_flooded_share"]), "F1 modelo": es(ext["model"]["f1"], 3),
           "ROC-AUC modelo": es(ext["model"]["roc_auc"], 3), "F1 PATRICOVA": es(refs["PATRICOVA"]["f1"], 3),
           "F1 SNCZI T=500": es(refs["SNCZI T=500"]["f1"], 3),
           "ROC-AUC SNCZI T=500": es(refs["SNCZI T=500"]["roc_auc"], 3), "F1 marcar todo": es(ext["mark_all_f1"], 3)}
    st.markdown("**Prueba externa: una zona que el modelo no ha visto**")
    st.dataframe(pd.DataFrame([row]), hide_index=True, width="stretch")
    st.caption(
        f"La Ribera nunca entra en el entrenamiento ni en el mapa: el modelo final la predice solo para medir si "
        f"generaliza (celdas a más de {es(ext['gap_m'])} m del área piloto). La validación por bloques de 1 km mide otra "
        "cosa: si el modelo rellena bien los huecos dentro de una zona que conoce, que es lo que hace el mapa."
    )


def field_depths(card: dict) -> None:
    """Puntos con calado medido en campo que caen en celdas Medio o Alto, frente a los mapas oficiales."""
    field = card["field_check"]
    st.markdown("**Comprobación con calados medidos en campo** (GVA: IVE, SGRT, UPV)")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric(f"Puntos con ≥ {es(field['min_depth_m'], 1)} m de agua", es(field["points"]),
              help=f"En {es(field['cells_with_points'])} celdas")
    f2.metric("En celdas Medio o Alto del modelo", f"{es(field['model_detected_pct'])} %")
    f3.metric("Dentro de PATRICOVA", f"{es(field['references_detected_pct']['PATRICOVA'])} %")
    f4.metric("Dentro del SNCZI T=500", f"{es(field['references_detected_pct']['SNCZI T=500'])} %")
    st.caption(
        "No es una validación independiente: la huella de la GVA, que forma parte de la etiqueta, cubre casi todos "
        "estos puntos. Sirve para ver cuánto de lo que se midió en campo quedaba dentro de los mapas oficiales."
    )


def importance_chart(card: dict) -> alt.Chart:
    """Barras horizontales de la importancia de cada variable, de mayor a menor, con los nombres completos."""
    importance = pd.DataFrame({
        "Variable": [FEATURE_LABELS.get(name, name) for name in card["feature_importance"]],
        "Importancia": list(card["feature_importance"].values()),
    })
    return alt.Chart(importance).mark_bar().encode(
        x=alt.X("Importancia:Q", title="Caída de PR-AUC al desordenar la variable",
                axis=alt.Axis(labelExpr="replace(format(datum.value, '.2f'), '.', ',')")),
        y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=400)),
        tooltip=["Variable", alt.Tooltip("Importancia:Q", format=".3f")],
    ).properties(height=320)


def model_details(card: dict) -> None:
    """ROC-AUC por grupo, falsos negativos por distrito, umbrales, importancia de las variables, fuentes y
    limitaciones."""
    st.markdown("**ROC-AUC por grupo de bloques** (cuánto varía según la zona)")
    by_fold = pd.DataFrame({MODEL_LABELS.get(name, name): [es(v, 3) for v in r["roc_auc_by_fold"]]
                            for name, r in card["models"].items()})
    st.dataframe(by_fold.set_axis([f"Grupo {i + 1}" for i in range(len(by_fold))]), width="stretch")

    st.markdown("**Falsos negativos por distrito o municipio** (muestra pequeña, interpretar con cautela)")
    misses = pd.DataFrame(card["false_negatives_by_district"])
    misses = misses.assign(fn_rate=misses["fn_rate"].map(pct)).rename(columns={
        "district": "Distrito o municipio", "flooded_cells": "Celdas inundadas", "false_negatives": "No detectadas",
        "fn_rate": "Parte no detectada",
    })
    st.dataframe(misses, hide_index=True, width="stretch")

    st.markdown(
        f"**Umbrales (fijados en validación):** Medio ≥ {es(card['thresholds']['medio'], 2)} (Medio + Alto encuentran "
        f"el {pct(TARGET_RECALL)} de lo inundado) · Alto ≥ {es(card['thresholds']['alto'], 2)} (concentra el "
        f"{pct(ALTO_RECALL)} de lo inundado, la parte de mayor probabilidad)"
    )
    st.markdown(f"**Variables utilizadas** (importancia por {card['feature_importance_method']})")
    st.altair_chart(importance_chart(card), width="stretch")

    st.markdown("**Fuentes de datos** (todas reales)")
    sources = services.get_data_sources().drop(columns=["id", "key"]).rename(columns={
        "name": "Organismo", "dataset_name": "Conjunto de datos", "url": "Descarga", "license": "Licencia",
        "file": "Fichero", "updated_at": "Actualizado",
    })
    st.dataframe(sources, hide_index=True, width="stretch",
                 column_config={"Descarga": st.column_config.LinkColumn(display_text="enlace")})
    st.markdown(
        "**Limitaciones:** el modelo aprende de un único episodio (la DANA del 29-10-2024): es un mapa de "
        "susceptibilidad aprendido de esa DANA, no una predicción del riesgo con cualquier lluvia. En otra cuenca que no "
        "ha visto (la Ribera) no supera a los mapas oficiales: la geología tendría sentido allí, pero en València y "
        "L'Horta casi no hay llanura de inundación de la que aprenderlo. La comparación con PATRICOVA y el SNCZI dentro "
        "del área favorece algo al modelo, que aprende de esta DANA; los mapas oficiales son anteriores. No usa la "
        "lluvia. La etiqueta une el satélite (Copernicus EMS), que no ve el agua entre edificios, y la huella oficial "
        "de la GVA, que cubre los cascos urbanos. Los arrozales y el agua permanente (lago de la Albufera, cauces; "
        "Corine Land Cover) se excluyen: no son suelo urbano y el satélite no distingue esa agua de la inundación. La "
        "validación por bloques de 1 km es algo optimista (con bloques de 3 km el F1 baja a 0,63, con y sin la altura "
        "sobre el cauce) y el F1 sube por la proporción de celdas inundadas (ver referencia trivial). La distancia y la "
        "altura sobre el cauce se diseñaron revisando este mismo episodio. Sin datos de drenaje, el "
        "modelo no representa la red de alcantarillado."
    )


main()
