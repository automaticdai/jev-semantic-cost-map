from jev_planner.events import initial_agvs, load_scenario, state_at


def scenario():
    return load_scenario("tests/fixtures/mini-scenario.yaml")


def test_scenario_loads_world_and_ticks():
    s = scenario()
    assert s.name == "mini"
    assert s.cells_per_tick == 2
    assert s.tick_count == 3
    assert s.world.width == 10


def test_notes_accumulate_across_ticks():
    s = scenario()
    agvs = initial_agvs(s.world)
    assert state_at(s, 0, agvs).notes["aisle-1"] == ()
    assert len(state_at(s, 1, agvs).notes["aisle-1"]) == 1
    notes = state_at(s, 2, agvs).notes["aisle-1"]
    assert len(notes) == 2
    assert "mopped" in notes[1]


def test_truth_defaults_to_clear_and_does_not_persist():
    s = scenario()
    agvs = initial_agvs(s.world)
    assert state_at(s, 0, agvs).truth == {"aisle-1": "clear", "aisle-2": "clear"}
    assert state_at(s, 1, agvs).truth["aisle-1"] == "blocked"
    assert state_at(s, 2, agvs).truth["aisle-1"] == "clear"


def test_clock_comes_from_the_tick():
    s = scenario()
    assert state_at(s, 1, initial_agvs(s.world)).clock == "13:10"


def test_initial_agvs_start_at_their_station_with_no_route():
    s = scenario()
    agvs = initial_agvs(s.world)
    assert agvs["agv-1"].cell == (0, 0)
    assert agvs["agv-1"].goal_station == "bay"
    assert agvs["agv-1"].committed_route == ()
    assert agvs["agv-1"].route_tick == -1


def test_states_are_independent_snapshots():
    s = scenario()
    agvs = initial_agvs(s.world)
    first = state_at(s, 1, agvs)
    state_at(s, 2, agvs)
    assert len(first.notes["aisle-1"]) == 1
