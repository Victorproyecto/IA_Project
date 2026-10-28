"""Dashboard DANA Predictor. Uso: streamlit run src/app/streamlit_app.py"""

import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import RISK_LEVELS  # noqa: E402
from src.data_access import geocoding, repository  # noqa: E402

# Paleta apta para daltonismo (azul-naranja-rojo); el nivel también va siempre en texto
COLORS = {"Bajo": "#2C7BB6", "Medio": "#E66101", "Alto": "#B2182B"}
OPACITY = {"Bajo": 0.15, "Medio": 0.55, "Alto": 0.7}
DISCLAIMER = (
    "**Herramienta informativa que no sustituye los avisos de las autoridades.** "
    "Muestra un riesgo estimado por celda de 250 m, no una predicción exacta del nivel del agua. "
    "Ante una emergencia, siga las indicaciones de Protección Civil, el 112 y ES-Alert."
)
KIND_LABEL = {"historical": "Histórico", "scenario": "Escenario simulado", "forecast": "Previsión AEMET"}

st.set_page_config(page_title="DANA Predictor", page_icon="🌧️", layout="wide")


@st.cache_data
def episodes() -> pd.DataFrame:
    return repository.list_episodes()


@st.cache_data
def hours(episode_id: int) -> list[pd.Timestamp]:
    return repository.get_episode_hours(episode_id)


@st.cache_data
def risk_map(episode_id: int, target_time):
    return repository.get_risk_map(episode_id, target_time)


@st.cache_data
def model_card() -> dict:
    return repository.get_model_card()


def episode_label(row: pd.Series) -> str:
    tag = "SIMULADO · " if row["is_simulated"] else ""
    return f"{row['name']} [{tag}{KIND_LABEL[row['kind']]}]"


def legend() -> None:
    chips = " ".join(
        f"<span style='display:inline-block;margin-right:12px'>"
        f"<span style='display:inline-block;width:14px;height:14px;background:{COLORS[level]};"
        f"border:1px solid #333;vertical-align:middle'></span> {level}</span>"
        for level in RISK_LEVELS
    )
    st.markdown(f"**Leyenda (nivel de riesgo estimado):** {chips}", unsafe_allow_html=True)


def build_map(data, center: tuple[float, float], zoom: int, selected_zone: int | None) -> folium.Map:
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    folium.GeoJson(
        data[["zone_id", "name", "district", "risk_level", "risk_probability", "altitude_avg",
              "imperviousness", "patricova_level", "geometry"]].assign(risk_level=lambda d: d["risk_level"].astype(str)),
        style_function=lambda f: {
            "fillColor": COLORS[f["properties"]["risk_level"]],
            "fillOpacity": OPACITY[f["properties"]["risk_level"]],
            "color": "#000000" if f["properties"]["zone_id"] == selected_zone else COLORS[f["properties"]["risk_level"]],
            "weight": 3 if f["properties"]["zone_id"] == selected_zone else 0.3,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "district", "risk_level", "risk_probability", "altitude_avg", "imperviousness", "patricova_level"],
            aliases=["Celda", "Distrito", "Nivel", "Probabilidad", "Altitud (m)", "Impermeable (%)", "PATRICOVA"],
            localize=True,
        ),
    ).add_to(fmap)
    return fmap


def zone_detail(zone_id: int, data) -> None:
    row = data[data["zone_id"] == zone_id]
    if row.empty:
        st.info("La ubicación está fuera del área piloto (ciudad de Valencia).")
        return
    row = row.iloc[0]
    st.subheader(f"{row['name']} · {row['district']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nivel de riesgo estimado", str(row["risk_level"]))
    c2.metric("Probabilidad estimada", f"{row['risk_probability']:.0%}")
    c3.metric("Estación de referencia", f"{row['station_id']} ({row['station_dist_m'] / 1000:.1f} km)")
    st.caption(
        f"Altitud {row['altitude_avg']:.1f} m · pendiente {row['slope_avg']:.2f} % · "
        f"impermeable {row['imperviousness']:.0f} % · PATRICOVA nivel {row['patricova_level'] or 'sin peligrosidad'}"
    )
    history = repository.get_zone_history(zone_id)
    st.markdown("**Antecedentes de inundación de la celda**")
    if history.empty:
        st.write("Sin inundaciones registradas en los episodios disponibles.")
    else:
        st.dataframe(history, hide_index=True, width="stretch")


def search_box() -> None:
    query = st.sidebar.text_input("Buscar dirección, calle o barrio", placeholder="Ej.: Plaza del Ayuntamiento")
    if not st.sidebar.button("Buscar", width="stretch") or not query:
        return
    places = repository.search_places(query)
    if not places.empty:
        place = places.iloc[0]
        st.session_state.update(center=(place["lat"], place["lon"]), zoom=15, search_label=f"{place['kind']} {place['name']}")
        st.session_state["zone"] = repository.find_zone(place["lat"], place["lon"])
        return
    result = geocoding.geocode(query)
    if result is None:
        st.sidebar.warning("No se ha encontrado esa ubicación dentro de Valencia.")
        return
    lat, lon, label = result
    st.session_state.update(center=(lat, lon), zoom=16, search_label=label, zone=repository.find_zone(lat, lon))


def main() -> None:
    st.title("DANA Predictor")
    st.caption("Mapa de riesgo de inundación urbana en Valencia ante episodios de lluvia intensa · prototipo académico")
    st.warning(DISCLAIMER, icon="⚠️")

    card = model_card()
    if card.get("trained_on_simulated_data"):
        st.info(
            "Esta versión usa **datos de muestra simulados** en las fuentes marcadas como «simulado» "
            "(pestaña *Modelo y datos*). Sirve para probar el sistema, no demuestra su capacidad predictiva real."
        )

    # Barra lateral: episodio, franja horaria y buscador
    eps = episodes()
    labels = {episode_label(r): r["episode_id"] for _, r in eps.iterrows()}
    st.sidebar.header("Consulta")
    episode_id = labels[st.sidebar.selectbox("Episodio o escenario", list(labels))]
    episode = eps.set_index("episode_id").loc[episode_id]

    # Los escenarios no tienen fecha real: se muestran en horas desde el inicio
    def time_label(t) -> str:
        if episode["kind"] == "scenario":
            return f"H+{int((pd.Timestamp(t) - episode['started_at']) / pd.Timedelta(hours=1))}"
        return pd.Timestamp(t).strftime("%d/%m/%Y %H:%M")

    show_peak = st.sidebar.checkbox("Ver el máximo del episodio", value=True)
    target_time = None
    if not show_peak:
        episode_hours = hours(episode_id)
        target_time = st.sidebar.select_slider(
            "Franja horaria", options=episode_hours, value=episode_hours[len(episode_hours) // 2], format_func=time_label
        )
    search_box()

    data = risk_map(episode_id, target_time)
    when = "máximo del episodio" if target_time is None else time_label(target_time)
    if episode["is_simulated"] and episode["kind"] == "scenario":
        st.markdown(f"#### {episode['name']} · **ESCENARIO SIMULADO** · {when}")
    else:
        st.markdown(f"#### {episode['name']} · {when}")

    counts = data["risk_level"].value_counts()
    cols = st.columns(4)
    for col, level in zip(cols, RISK_LEVELS):
        col.metric(f"Celdas en nivel {level}", int(counts.get(level, 0)))
    cols[3].metric("Modelo", card["model_version"])

    tab_map, tab_table, tab_history, tab_model = st.tabs(["Mapa", "Tabla de zonas", "Consulta histórica", "Modelo y datos"])

    with tab_map:
        legend()
        if "search_label" in st.session_state:
            st.caption(f"Ubicación buscada: {st.session_state['search_label']}")
        center = st.session_state.get("center", (39.4699, -0.3763))
        fmap = build_map(data, center, st.session_state.get("zoom", 13), st.session_state.get("zone"))
        clicked = st_folium(fmap, height=560, width="stretch", returned_objects=["last_clicked"])
        if clicked and clicked.get("last_clicked"):
            point = clicked["last_clicked"]
            st.session_state["zone"] = repository.find_zone(point["lat"], point["lng"])
        if st.session_state.get("zone"):
            zone_detail(st.session_state["zone"], data)
        else:
            st.caption("Pulse una celda del mapa o use el buscador para ver su detalle.")

    with tab_table:
        st.markdown("Alternativa accesible al mapa: zonas ordenables por nivel y probabilidad.")
        table = data[["name", "district", "risk_level", "risk_probability", "altitude_avg", "imperviousness", "patricova_level"]]
        table = table.sort_values(["risk_level", "risk_probability"], ascending=False).rename(columns={
            "name": "Celda", "district": "Distrito", "risk_level": "Nivel", "risk_probability": "Probabilidad",
            "altitude_avg": "Altitud (m)", "imperviousness": "Impermeable (%)", "patricova_level": "PATRICOVA",
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown("**Resumen por distrito**")
        st.dataframe(repository.get_district_summary(episode_id, target_time), hide_index=True, width="stretch")
        st.download_button("Descargar CSV", table.to_csv(index=False).encode(), file_name=f"riesgo_episodio_{episode_id}.csv")

    with tab_history:
        district = st.selectbox("Distrito", sorted(repository.get_zones()["district"].dropna().unique()))
        history = repository.get_district_history(district)
        if history.empty:
            st.write("Sin inundaciones registradas en este distrito.")
        else:
            st.dataframe(history.rename(columns={
                "name": "Episodio", "started_at": "Inicio", "celdas_inundadas": "Celdas inundadas",
                "fraccion_afectada": "Fracción afectada",
            }), hide_index=True, width="stretch")

    with tab_model:
        model_tab(card)


def model_tab(card: dict) -> None:
    st.markdown(f"**Versión:** {card['model_version']} · **modelo:** {card['chosen_model']} · entrenado {card['trained_at']}")
    criteria = card["success_criteria"]
    val = card["models"][card["chosen_model"]]["validation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (validación)", f"{val['recall']:.2f}", help=f"Objetivo ≥ {criteria['recall_target']}")
    c2.metric("F1 (validación)", f"{val['f1']:.2f}", help=f"Objetivo ≥ {criteria['f1_target']}")
    c3.metric("ROC-AUC", f"{val['roc_auc']:.3f}")
    c4.metric("PR-AUC", f"{val['pr_auc']:.3f}")

    rows = []
    for name, result in card["models"].items():
        rows.append({"modelo": name, "conjunto": "validación", **{k: result["validation"][k] for k in ("recall", "f1", "precision", "pr_auc")}})
        rows.append({"modelo": name, "conjunto": "retrospectiva", **{k: result["test"][k] for k in ("recall", "f1", "precision", "pr_auc")}})
    for name, result in card["baselines"].items():
        rows.append({"modelo": f"referencia: {name}", "conjunto": "validación", **{k: result["validation"][k] for k in ("recall", "f1", "precision", "pr_auc")}})
    st.markdown("**Comparativa de modelos y referencias**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    retro = card["retrospective_test"]
    st.markdown(f"**Prueba retrospectiva:** {retro['episode']} (excluido del entrenamiento)")
    st.json(retro["metrics"]["confusion_matrix"], expanded=False)
    st.markdown("**Falsos negativos por distrito** (muestra pequeña, interpretar con cautela)")
    st.dataframe(pd.DataFrame(retro["false_negatives_by_district"]), hide_index=True, width="stretch")

    st.markdown(
        f"**Umbrales (fijados en validación):** Medio ≥ {card['thresholds']['medio']:.2f} · "
        f"Alto ≥ {card['thresholds']['alto']:.2f}"
    )
    st.markdown("**Variables utilizadas**")
    st.dataframe(pd.DataFrame(card["feature_importance"].items(), columns=["variable", "peso"]), hide_index=True)

    st.markdown("**Fuentes de datos**")
    st.dataframe(repository.get_data_sources().drop(columns=["key"]), hide_index=True, width="stretch")
    st.markdown(
        "**Limitaciones:** no predice el nivel de agua por calle ni garantiza que la inundación ocurra; "
        "la lluvia se asigna desde la estación más cercana y la previsión de AEMET es municipal; "
        "PATRICOVA es una peligrosidad modelada, no lo observado; sin datos de drenaje el modelo no "
        "representa la red de alcantarillado."
    )


main()
