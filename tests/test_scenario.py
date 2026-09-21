import pytest

from jev_planner.baseline import DEFAULT_RULES, baseline_costs
from jev_planner.events import initial_agvs, load_scenario, state_at
from jev_planner.planner import plan_single
from tests.helpers import uniform_layer

SCENARIO = "config/scenarios/night-shift.yaml"


def scenario():
    return load_scenario(SCENARIO)


def test_the_floor_has_fourteen_zones_and_no_overlaps():
    assert len(scenario().world.zones) == 14      # load_world raises on overlap


def test_the_shift_runs_twelve_ticks():
    assert scenario().tick_count == 12


def test_every_station_is_reachable_from_every_other():
    s = scenario()
    layer = uniform_layer(s.world)
    cells = [station.cell for station in s.world.stations.values()]
    for start in cells:
        for goal in cells:
            if start == goal:
                continue
            assert plan_single(s.world, layer, "probe", start, goal) is not None


def test_no_agv_starts_on_a_wall():
    s = scenario()
    for agv in initial_agvs(s.world).values():
        assert s.world.passable(agv.cell)


def test_the_baseline_will_not_release_the_mopped_spill():
    """Tick 3 mopped it and the truth says clear. The rule list cannot tell."""
    s = scenario()
    ws = state_at(s, 3, initial_agvs(s.world))
    assert ws.truth["aisle-3"] == "clear"
    assert baseline_costs(s.world, ws, DEFAULT_RULES)["aisle-3"].blocked is True


def test_the_baseline_cannot_see_the_stock_counter_in_the_lane():
    """Tick 4 puts a person in the aisle-2 lane in words no rule matches."""
    s = scenario()
    ws = state_at(s, 4, initial_agvs(s.world))
    assert ws.truth["aisle-2"] == "blocked"
    cost = baseline_costs(s.world, ws, DEFAULT_RULES)["aisle-2"]
    assert cost.blocked is False
    assert cost.multiplier == pytest.approx(1.0)


def test_the_baseline_does_catch_the_uncleaned_leak():
    """The baseline is a fair opponent: tick 7 is a case it gets right."""
    s = scenario()
    ws = state_at(s, 7, initial_agvs(s.world))
    assert ws.truth["aisle-5"] == "blocked"
    assert baseline_costs(s.world, ws, DEFAULT_RULES)["aisle-5"].blocked is True


def test_a_blocked_junction_never_disconnects_the_floor():
    """junction-4 is blocked at ticks 9 and 10; traffic must still route north."""
    s = scenario()
    layer = uniform_layer(s.world, blocked=("junction-4",))
    assert plan_single(s.world, layer, "probe", (2, 12), (73, 34)) is not None
