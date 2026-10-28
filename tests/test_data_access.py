from src.config import RISK_LEVELS
from src.data_access import repository


def test_risk_map_has_one_row_per_zone(built_data):
    episode_id = int(repository.list_episodes().iloc[0]["episode_id"])
    risk_map = repository.get_risk_map(episode_id)
    assert len(risk_map) == len(repository.get_zones())
    assert risk_map.crs.to_string() == "EPSG:4326"


def test_predictions_at_a_given_hour(built_data):
    episode_id = int(repository.list_episodes().iloc[0]["episode_id"])
    hour = repository.get_episode_hours(episode_id)[10]
    predictions = repository.get_predictions(episode_id, hour)
    assert (predictions["target_time"] == hour).all()
    assert predictions["zone_id"].is_unique


def test_find_zone_inside_and_outside_pilot(built_data):
    assert repository.find_zone(39.4697, -0.3774) is not None
    assert repository.find_zone(40.4168, -3.7038) is None


def test_zone_history_and_district_summary(built_data):
    episodes = repository.list_episodes()
    historical = int(episodes[episodes["kind"] == "historical"].iloc[0]["episode_id"])
    summary = repository.get_district_summary(historical)
    assert set(RISK_LEVELS) <= set(summary.columns)
    flooded_zone = int(repository.get_risk_map(historical).sort_values("risk_probability").iloc[-1]["zone_id"])
    assert {"episode", "severity", "source"} <= set(repository.get_zone_history(flooded_zone).columns)


def test_search_places_by_district_name(built_data):
    result = repository.search_places("ciutat")
    assert "Ciutat Vella" in result["name"].tolist()
