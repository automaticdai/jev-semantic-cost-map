import pytest

from jev_costmap.baseline import DEFAULT_RULES, baseline_costs, load_rules
from jev_costmap.costs import DEFAULT_WEIGHTS, load_weights
from jev_costmap.events import initial_agvs, load_scenario, state_at
from jev_costmap.planner import plan_single
from tests.helpers import uniform_layer

SCENARIO = "config/scenarios/night-shift.yaml"

AISLE_IDS = ("aisle-1", "aisle-2", "aisle-3", "aisle-4", "aisle-5", "aisle-6")


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


def test_each_aisle_actually_connects_the_north_and_south_corridors():
    """test_every_station_is_reachable_from_every_other cannot see a sealed
    aisle: dock-A, dock-B, stage and charge all sit in bays or staging that
    touch the full-width north and south cross corridors directly, so every
    station-to-station route can go around a blocked aisle via that corridor
    "highway" without ever entering it. Blocking every OTHER aisle removes
    that detour, so each check below fails if and only if the aisle under
    test is itself sealed. The final assertion proves the corridors alone
    cannot connect north to south, which is what makes the per-aisle checks
    meaningful rather than vacuous.
    """
    s = scenario()
    world = s.world
    for aisle_id in AISLE_IDS:
        other_aisles = tuple(a for a in AISLE_IDS if a != aisle_id)
        layer = uniform_layer(world, blocked=other_aisles)
        x = world.zones[aisle_id].x + 1
        north, south = (x, 6), (x, 43)
        assert plan_single(world, layer, "probe", north, south) is not None, aisle_id

    all_blocked = uniform_layer(world, blocked=AISLE_IDS)
    x = world.zones[AISLE_IDS[0]].x + 1
    assert plan_single(world, all_blocked, "probe", (x, 6), (x, 43)) is None


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


def test_the_shipped_config_is_what_the_fairness_tests_above_actually_tested():
    """Every baseline-fairness test above passes DEFAULT_RULES / DEFAULT_WEIGHTS
    directly, not config/rules.yaml or config/weights.yaml -- but the CLI's
    `run` command reads those files (`load_rules(args.rules)`,
    `load_weights(args.weights)`). The two are byte-identical today, so the
    fairness tests happen to describe the shipped comparison, but nothing
    enforces that. Someone could add `stalled` to config/rules.yaml -- the
    single most defensible edit to that file, since a standing-rule judgment
    call for a stalled AGV is exactly what the baseline currently misses --
    and every fairness test above would keep passing while the run the CLI
    actually produces silently stopped matching what they tested. This
    assertion is what makes those tests meaningful statements about the
    shipped config rather than about a config that merely used to match it.
    """
    assert tuple(load_rules("config/rules.yaml")) == DEFAULT_RULES
    assert load_weights("config/weights.yaml") == DEFAULT_WEIGHTS


def test_a_blocked_junction_never_disconnects_the_floor():
    """junction-4 is blocked at ticks 9 and 10; traffic must still route north."""
    s = scenario()
    layer = uniform_layer(s.world, blocked=("junction-4",))
    assert plan_single(s.world, layer, "probe", (2, 12), (73, 34)) is not None
