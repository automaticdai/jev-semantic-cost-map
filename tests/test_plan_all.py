import pytest

from jev_planner.planner import PlanRequest, plan_all
from tests.helpers import grid_world, uniform_layer


def assert_no_conflicts(plans):
    """No two AGVs share a cell at a timestep, and none swap through each other."""
    paths = {p.agv: p.path for p in plans.values()}
    span = max(len(p) for p in paths.values())
    for t in range(span):
        at = {}
        for agv, path in paths.items():
            cell = path[min(t, len(path) - 1)]
            assert cell not in at, f"{agv} and {at[cell]} share {cell} at t={t}"
            at[cell] = agv
        if t == 0:
            continue
        for a, pa in paths.items():
            for b, pb in paths.items():
                if a >= b:
                    continue
                a_prev, a_now = pa[min(t - 1, len(pa) - 1)], pa[min(t, len(pa) - 1)]
                b_prev, b_now = pb[min(t - 1, len(pb) - 1)], pb[min(t, len(pb) - 1)]
                assert not (a_prev == b_now and b_prev == a_now), f"{a} and {b} swapped at t={t}"


def test_the_edge_rule_holds_end_to_end():
    """`Reservations.edge` is unit-tested above; this is the behavioural check
    that plan_all never produces a swap, which no vertex check would catch."""
    w = grid_world(5, 2)
    plans = plan_all(
        w,
        uniform_layer(w),
        [PlanRequest("agv-1", 1, (0, 0), (4, 0)), PlanRequest("agv-2", 2, (4, 0), (0, 1))],
    )
    assert_no_conflicts(plans)


def test_two_agvs_pass_each_other_in_a_two_wide_lane():
    w = grid_world(5, 2)
    plans = plan_all(
        w,
        uniform_layer(w),
        [
            PlanRequest("agv-1", 1, (0, 0), (4, 0)),
            PlanRequest("agv-2", 2, (4, 0), (0, 0)),
        ],
    )
    assert not plans["agv-1"].deferred
    assert not plans["agv-2"].deferred
    assert_no_conflicts(plans)


def test_the_highest_priority_agv_gets_the_direct_path():
    w = grid_world(5, 2)
    plans = plan_all(
        w,
        uniform_layer(w),
        [
            PlanRequest("agv-2", 2, (4, 0), (0, 0)),
            PlanRequest("agv-1", 1, (0, 0), (4, 0)),
        ],
    )
    assert plans["agv-1"].path == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))


def test_a_starved_agv_is_deferred_and_holds_position():
    # agv-1 parks in the middle of a one-wide lane, so agv-2 has no route at all.
    w = grid_world(5, 1)
    plans = plan_all(
        w,
        uniform_layer(w),
        [
            PlanRequest("agv-1", 1, (0, 0), (2, 0)),
            PlanRequest("agv-2", 2, (4, 0), (0, 0)),
        ],
    )
    assert plans["agv-1"].deferred is False
    assert plans["agv-2"].deferred is True
    assert plans["agv-2"].path == ((4, 0),)
    assert plans["agv-2"].cost == pytest.approx(0.0)


def test_every_requested_agv_appears_in_the_result():
    w = grid_world(5, 1)
    requests = [PlanRequest("agv-1", 1, (0, 0), (4, 0)), PlanRequest("agv-2", 2, (4, 0), (0, 0))]
    assert set(plan_all(w, uniform_layer(w), requests)) == {"agv-1", "agv-2"}
