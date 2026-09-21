import json

from jev_planner.events import AgvState, initial_agvs, load_scenario, state_at
from jev_planner.state import build_state


def built(tick: int = 2):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    agvs = initial_agvs(s.world)
    agvs["agv-1"] = AgvState("agv-1", (4, 1), "bay", ("aisle-1", "aisle-2"), tick - 1)
    return build_state(s.world, state_at(s, tick, agvs))


def test_truth_labels_never_reach_the_model():
    blob = json.dumps(built())
    for label in ("blocked", "avoid", "truth"):
        assert label not in blob


def test_top_level_keys_are_exactly_these():
    assert set(built()) == {"shift", "zones", "agvs"}


def test_zone_carries_description_and_accumulated_notes():
    zone = built()["zones"]["aisle-1"]
    assert zone["name"] == "Aisle 1"
    assert zone["kind"] == "travel lane"
    assert "pallet racking" in zone["description"]
    assert len(zone["notes"]) == 2
    assert "mopped" in zone["notes"][1]


def test_agv_reports_its_zone_and_the_tick_its_route_came_from():
    agv = built(tick=2)["agvs"]["agv-1"]
    assert agv["position"] == "aisle-1"
    assert agv["committed_route"] == ["aisle-1", "aisle-2"]
    assert agv["route_from_tick"] == 1


def test_state_is_json_serialisable():
    json.dumps(built())
