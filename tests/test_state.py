import json

from jev_costmap.events import AgvState, initial_agvs, load_scenario, state_at
from jev_costmap.state import build_state


def built(tick: int = 2):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    agvs = initial_agvs(s.world)
    agvs["agv-1"] = AgvState("agv-1", (4, 1), "bay", ("aisle-1", "aisle-2"), tick - 1)
    return build_state(s.world, state_at(s, tick, agvs))


def test_truth_labels_never_reach_the_model():
    # Test tick 1 where truth["aisle-1"] == "blocked" exists
    state_tick1 = built(tick=1)
    blob_tick1 = json.dumps(state_tick1)
    for label in ("blocked", "avoid", "truth"):
        assert label not in blob_tick1, f"Label '{label}' leaked at tick 1"

    # Structural assertion: zones must have exactly these keys
    for zone_id, zone in state_tick1["zones"].items():
        assert set(zone.keys()) == {"name", "kind", "description", "notes"}, \
            f"Zone {zone_id} has unexpected keys: {set(zone.keys())}"

    # Structural assertion: agvs must have exactly these keys
    for agv_id, agv in state_tick1["agvs"].items():
        assert set(agv.keys()) == {"position", "committed_route", "route_from_tick"}, \
            f"AGV {agv_id} has unexpected keys: {set(agv.keys())}"

    # Test tick 2 where truth["aisle-1"] == "clear" (label absent case)
    state_tick2 = built(tick=2)
    blob_tick2 = json.dumps(state_tick2)
    for label in ("blocked", "avoid", "truth"):
        assert label not in blob_tick2, f"Label '{label}' leaked at tick 2"

    # Structural assertion for tick 2 as well
    for zone_id, zone in state_tick2["zones"].items():
        assert set(zone.keys()) == {"name", "kind", "description", "notes"}, \
            f"Zone {zone_id} has unexpected keys: {set(zone.keys())}"

    for agv_id, agv in state_tick2["agvs"].items():
        assert set(agv.keys()) == {"position", "committed_route", "route_from_tick"}, \
            f"AGV {agv_id} has unexpected keys: {set(agv.keys())}"


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


def test_agv_outside_any_named_zone_uses_fallback_position():
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    agvs = initial_agvs(s.world)
    # Cell (3, 3) is on the wall row and belongs to no zone
    agvs["agv-1"] = AgvState("agv-1", (3, 3), "bay", ("aisle-1",), 0)
    state = build_state(s.world, state_at(s, 1, agvs))
    agv = state["agvs"]["agv-1"]
    assert agv["position"] == "outside any named zone"
