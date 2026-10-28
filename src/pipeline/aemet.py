"""Cliente mínimo de AEMET OpenData para la previsión horaria de Valencia.

AEMET responde en dos pasos: la primera llamada devuelve la URL ("datos") con el contenido real.
La previsión es municipal, así que se aplica igual a todas las celdas (Fase 1, 9.4).
"""

import json

import pandas as pd
import requests

from src.config import AEMET_API_KEY, AEMET_VALENCIA_MUNICIPIO, RAW_DIR

BASE_URL = "https://opendata.aemet.es/opendata/api"


def _fetch(endpoint: str) -> list[dict]:
    response = requests.get(f"{BASE_URL}{endpoint}", params={"api_key": AEMET_API_KEY}, timeout=30)
    response.raise_for_status()
    data = requests.get(response.json()["datos"], timeout=30)
    data.raise_for_status()
    return data.json()


def _number(value: str) -> float:
    # "Ip" = precipitación inapreciable
    return 0.0 if value in ("", "Ip") else float(value)


def hourly_forecast() -> pd.DataFrame:
    """Previsión horaria: observed_at, precipitation_mm, temperature_c. Guarda el JSON bruto en data/raw/."""
    payload = _fetch(f"/prediccion/especifica/municipio/horaria/{AEMET_VALENCIA_MUNICIPIO}")
    (RAW_DIR / "aemet_forecast.json").write_text(json.dumps(payload, ensure_ascii=False))

    rows = []
    for day in payload[0]["prediccion"]["dia"]:
        date = pd.Timestamp(day["fecha"]).normalize()
        temperature = {t["periodo"]: _number(t["value"]) for t in day.get("temperatura", [])}
        for p in day.get("precipitacion", []):
            rows.append(
                {
                    "observed_at": date + pd.Timedelta(hours=int(p["periodo"])),
                    "precipitation_mm": _number(p["value"]),
                    "temperature_c": temperature.get(p["periodo"]),
                }
            )
    return pd.DataFrame(rows).sort_values("observed_at").reset_index(drop=True)
