"""Geocodificación de direcciones con Nominatim (OpenStreetMap).

La dirección solo se usa para centrar el mapa: no se guarda ni se registra (Fase 1, 9.1).
"""

import requests

from src.config import CITY_BBOX

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "DANA-Predictor/0.1 (prototipo academico)"


def geocode(address: str) -> tuple[float, float, str] | None:
    """Devuelve (lat, lon, nombre) del primer resultado dentro de Valencia, o None."""
    lon_min, lat_min, lon_max, lat_max = CITY_BBOX
    try:
        response = requests.get(
            NOMINATIM_URL,
            params={
                "q": f"{address}, València",
                "format": "json",
                "limit": 1,
                "viewbox": f"{lon_min},{lat_max},{lon_max},{lat_min}",
                "bounded": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()
    except requests.RequestException:
        return None
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"]), results[0]["display_name"]
