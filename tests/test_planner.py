import pytest

from jev_costmap.planner import HORIZON, Plan, Reservations, plan_single
from tests.helpers import grid_world, uniform_layer


def test_reservations_record_every_vertex():
    r = Reservations()
    r.add(((0, 0), (1, 0), (2, 0)))
    assert r.vertex((1, 0), 1)
    assert not r.vertex((1, 0), 0)


def test_a_parked_agv_holds_its_goal_to_the_horizon():
    r = Reservations()
    r.add(((0, 0), (1, 0)))
    assert r.vertex((1, 0), 1)
    assert r.vertex((1, 0), 50)
    assert r.vertex((1, 0), HORIZON)


def test_reservations_detect_a_swap():
    r = Reservations()
    r.add(((4, 0), (3, 0), (2, 0)))
    assert r.edge((3, 0), (4, 0), 1)      # we would move east into their westward move
    assert not r.edge((3, 0), (4, 0), 2)


def test_a_clear_lane_gives_the_direct_path():
    w = grid_world(5, 1)
    plan = plan_single(w, uniform_layer(w), "agv-1", (0, 0), (4, 0))
    assert plan.path == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    assert plan.cost == pytest.approx(4.0)
    assert plan.deferred is False


def test_an_expensive_lane_is_detoured_around():
    w = grid_world(5, 2)
    layer = uniform_layer(w, multipliers={"row-1": 10.0})
    plan = plan_single(w, layer, "agv-1", (0, 1), (4, 1))
    assert (2, 0) in plan.path
    assert plan.cost < 4 * 10.0


def test_a_blocked_zone_is_never_entered():
    w = grid_world(5, 2)
    layer = uniform_layer(w, blocked=("row-0",))
    plan = plan_single(w, layer, "agv-1", (0, 1), (4, 1))
    assert all(cell[1] == 1 for cell in plan.path)


def test_a_blocked_goal_has_no_plan():
    w = grid_world(5, 2)
    layer = uniform_layer(w, blocked=("row-0",))
    assert plan_single(w, layer, "agv-1", (0, 1), (4, 0)) is None


def test_an_agv_starting_in_a_newly_blocked_zone_can_still_leave():
    w = grid_world(5, 2)
    layer = uniform_layer(w, blocked=("row-0",))
    plan = plan_single(w, layer, "agv-1", (0, 0), (4, 1))
    assert plan is not None
    assert plan.path[0] == (0, 0)
    assert all(cell[1] == 1 for cell in plan.path[1:])


def test_a_reserved_cell_makes_the_agv_wait():
    w = grid_world(5, 1)
    r = Reservations()
    r._vertex.add(((1, 0), 1))       # a bare vertex hold, no parked path
    plan = plan_single(w, uniform_layer(w), "agv-1", (0, 0), (4, 0), reservations=r)
    assert plan.path[1] == (0, 0)
    assert len(plan.path) == 6
    assert plan.cost == pytest.approx(4.0 + 0.5)


def test_an_agv_parked_across_a_one_wide_lane_blocks_it_for_good():
    """The other AGV stops in the middle and stays there, so there is no route
    and no amount of waiting produces one. Reserving a parked goal only to the
    end of its path instead of to the horizon would wrongly find a plan here."""
    w = grid_world(5, 1)
    r = Reservations()
    r.add(((0, 0), (1, 0), (2, 0)))
    assert plan_single(w, uniform_layer(w), "agv-2", (4, 0), (0, 0), reservations=r) is None


def test_two_agvs_can_pass_in_a_two_wide_lane():
    w = grid_world(5, 2)
    r = Reservations()
    r.add(((4, 0), (3, 0), (2, 0), (1, 0), (0, 0)))
    plan = plan_single(w, uniform_layer(w), "agv-2", (0, 1), (4, 1), reservations=r)
    assert plan is not None


def test_the_horizon_is_respected():
    w = grid_world(5, 1)
    assert plan_single(w, uniform_layer(w), "agv-1", (0, 0), (4, 0), horizon=2) is None


def test_a_live_search_detects_an_oncoming_swap():
    """agv-2's direct first step (1,0) -> (2,0) at t=1 is a head-on swap
    against the reserved westbound move (2,0) -> (1,0) arriving at t=1.
    Crucially (2,0) is not vertex-reserved at t=1, so only the edge rule can
    reject this move. A reversed comparison inside Reservations.edge, or a
    call site that passes (nxt, cell, t + 1) instead of (cell, nxt, t + 1),
    would fail to catch it and let the plan pass straight through the
    oncoming AGV via the direct, cheaper route."""
    w = grid_world(4, 2)
    r = Reservations()
    r.add(((2, 0), (1, 0), (0, 0)))
    plan = plan_single(w, uniform_layer(w), "agv-2", (1, 0), (3, 0), reservations=r)
    assert plan is not None
    assert plan.path[1] != (2, 0)
