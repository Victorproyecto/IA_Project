from src.config import RISK_LEVELS
from src.backend import services


def test_risk_map_has_one_row_per_zone(built_data):
    risk_map = services.get_risk_map()
    assert len(risk_map) == len(services.get_zones())
    assert risk_map.crs.to_string() == "EPSG:4326"
    assert risk_map["observed"].notna().all()


def test_area_outline_has_one_line_per_comarca(built_data):
    from src.config import PILOT_COMARCAS

    outline = services.get_area_outline()
    assert {feature["properties"]["comarca"] for feature in outline["features"]} == set(PILOT_COMARCAS)
    assert all(feature["geometry"]["type"].endswith("LineString") for feature in outline["features"])


def test_find_zone_inside_and_outside_pilot(built_data):
    zone = services.get_zones_web().iloc[0]
    center = zone.geometry.centroid
    assert services.find_zone(center.y, center.x) == zone["zone_id"]
    assert services.find_zone(40.4168, -3.7038) is None  # Madrid


def test_zone_history_and_district_summary(built_data):
    summary = services.get_district_summary()
    assert set(RISK_LEVELS) | {"inundadas_dana"} <= set(summary.columns)
    flooded = services.get_risk_map().dropna(subset=["severity"]).iloc[0]
    history = services.get_zone_history(int(flooded["zone_id"]))
    assert {"episode", "severity", "source"} <= set(history.columns)
    assert len(history) == 1


def test_search_places_by_part_of_a_district_name(built_data):
    district = services.get_zones()["district"].dropna().iloc[0]
    result = services.search_places(district[:5].lower())
    assert district.lower() in result["name"].str.lower().tolist()


def test_places_have_a_center_inside_the_pilot(built_data):
    """Cada municipio, distrito y barrio del buscador tiene su centro (WGS84) dentro del área piloto."""
    from rasterio.warp import transform_bounds

    from src.config import CRS, CRS_WEB, PILOT_EXTENT

    places = services.list_places()
    assert set(places["kind"]) == {"Municipio", "Distrito", "Barrio"}
    assert places["name"].is_unique
    west, south, east, north = transform_bounds(CRS, CRS_WEB, *PILOT_EXTENT)
    assert places["lon"].between(west, east).all() and places["lat"].between(south, north).all()


def test_map_grid_codes_match_zones_levels_and_flooding(built_data):
    """La cuadrícula que recibe el navegador: una celda por zona, con su nivel y su gravedad observada."""
    import base64

    import numpy as np
    import pandas as pd

    from src.config import FLOOD_EVENTS_FILE, SEVERITY_ORDER

    grid = services.get_map_grid()
    codes = np.frombuffer(base64.b64decode(grid["codes"]), dtype=np.uint8)
    assert len(codes) == grid["rows"] * grid["cols"]
    zones = services.get_zones().merge(services.get_predictions()[["zone_id", "risk_level"]], on="zone_id")
    cell = zones["zone_id"].to_numpy() - 1  # zone_id = row * cols + col + 1
    assert (codes != 0).sum() == len(zones)
    assert ((codes[cell] & 3) == zones["risk_level"].cat.codes.to_numpy() + 1).all()
    events = pd.read_parquet(FLOOD_EVENTS_FILE).merge(zones[["zone_id"]], on="zone_id")
    severity = codes[events["zone_id"].to_numpy() - 1] >> services.SEVERITY_SHIFT
    assert (severity == events["severity"].map(SEVERITY_ORDER.index).to_numpy() + 1).all()
    assert (codes >> services.SEVERITY_SHIFT != 0).sum() == len(events)
