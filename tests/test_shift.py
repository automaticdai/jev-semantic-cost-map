import datetime as dt

import httpx2
import pytest
from typesafe_sdk import SystemOneResponse, TypeSafeAPIError

from jev_costmap.baseline import DEFAULT_RULES
from jev_costmap.costs import DEFAULT_WEIGHTS
from jev_costmap.events import AgvState, initial_agvs, load_scenario
from jev_costmap.judge import Judge
from jev_costmap.planner import Plan
from jev_costmap.runs import RunDir
from jev_costmap.shift import ShiftAborted, advance, run_shift, run_tick


class CalmClient:
    """Answers every question with 'nothing going on here'."""

    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kwargs):
        self.calls += 1
        answers = {}
        for qid in questions:
            if qid.endswith(".offlimits"):
                answers[qid] = {"type": "noul", "noul": 0.02}
            else:
                answers[qid] = {
                    "type": "score", "score": 0.0, "confidence": 0.95,
                    "legend": {0: "a", 1: "b", 2: "c", 3: "d"},
                    "probabilities": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                }
        return SystemOneResponse.model_validate(
            {"model": "jev-1.13.0", "usage": {"input_tokens": 500}, "answers": answers}
        )


def scenario():
    return load_scenario("tests/fixtures/mini-scenario.yaml")


def test_advance_moves_the_agv_and_records_the_route_tick():
    s = scenario()
    agvs = initial_agvs(s.world)
    plans = {"agv-1": Plan("agv-1", ((0, 0), (1, 0), (2, 0), (3, 0)), 3.0, False)}
    moved = advance(s, agvs, plans, tick=0)
    assert moved["agv-1"].cell == (2, 0)          # cells_per_tick is 2
    assert moved["agv-1"].route_tick == 0
    assert moved["agv-1"].committed_route == ("aisle-1",)


def test_a_deferred_agv_does_not_move():
    s = scenario()
    agvs = initial_agvs(s.world)
    plans = {"agv-1": Plan("agv-1", ((0, 0),), 0.0, True)}
    assert advance(s, agvs, plans, tick=0)["agv-1"].cell == (0, 0)


def test_run_shift_writes_one_file_per_kind_per_tick(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    judge = Judge(client=CalmClient(), cache_path=tmp_path / "c.json")
    run_shift(s, judge, run, DEFAULT_WEIGHTS, DEFAULT_RULES)

    assert run.ticks == [0, 1, 2]
    for tick in run.ticks:
        for kind in ("states", "questions", "answers", "costs", "plans", "truth"):
            run.read_tick(kind, tick)


def test_the_manifest_records_the_model_tokens_and_weights(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    m = run.read_manifest()
    assert m["model"] == "jev-1.13.0"
    assert m["input_tokens"] == 1500
    assert m["weights"] == {"people": 6.0, "damage": 3.0, "delay": 1.5}
    assert all(t["status"] == "ok" for t in m["ticks"])


def test_the_manifest_records_a_git_sha(tmp_path):
    """The spec requires the manifest to carry the git sha the run was produced
    under. It may be None outside a git checkout, but the key must exist."""
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    manifest = run.read_manifest()
    assert "git_sha" in manifest
    assert manifest["git_sha"] is None or isinstance(manifest["git_sha"], str)


def test_the_manifest_records_the_scenario_path(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert run.read_manifest()["scenario_path"].endswith("mini-scenario.yaml")


def test_stored_plans_carry_both_sources(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    plans = run.read_tick("plans", 1)
    assert set(plans) == {"jev", "baseline"}
    assert "agv-1" in plans["jev"]


def test_the_second_tick_tells_jev_the_route_from_the_first(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert run.read_tick("states", 1)["agvs"]["agv-1"]["route_from_tick"] == 0


class FailingClient(CalmClient):
    """Answers tick 0, then fails the way the SDK does once retries are spent."""

    def system_one(self, state, questions, **kwargs):
        if self.calls >= 1:
            # The installed SDK's TypeSafeAPIError requires status/body/headers
            # (the brief's single-string constructor call predates this signature).
            raise TypeSafeAPIError(503, "upstream unavailable", httpx2.Headers())
        return super().system_one(state, questions, **kwargs)


def test_a_failed_tick_aborts_the_run_and_keeps_what_was_written(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    judge = Judge(client=FailingClient(), cache_path=tmp_path / "c.json")
    with pytest.raises(ShiftAborted):
        run_shift(s, judge, run, DEFAULT_WEIGHTS, DEFAULT_RULES)

    assert run.ticks == [0]                       # tick 0's artifacts survived
    entries = run.read_manifest()["ticks"]
    assert entries[0]["status"] == "ok"
    assert entries[1]["status"] == "failed"
    assert "503" in entries[1]["error"]


def test_a_failure_never_falls_back_to_the_baseline(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    with pytest.raises(ShiftAborted):
        run_shift(s, Judge(client=FailingClient(), cache_path=tmp_path / "c.json"),
                  run, DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert 1 not in run.ticks


def test_truth_is_stored_for_scoring_but_not_in_the_state(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert run.read_tick("truth", 1)["aisle-1"] == "blocked"
    assert "blocked" not in str(run.read_tick("states", 1))


def test_both_planners_plan_from_identical_agv_positions():
    """Both cost models must be scored from the same situation; if one planned
    from positions the other had already advanced past, every comparison in
    the report would be meaningless."""
    s = scenario()
    agvs = {
        "agv-1": AgvState(
            id="agv-1", cell=(5, 1), goal_station="bay",
            committed_route=(), route_tick=-1,
        )
    }
    judge = Judge(client=CalmClient(), cache_path=None)
    result = run_tick(s, judge, agvs, tick=0, weights=DEFAULT_WEIGHTS, rules=DEFAULT_RULES)

    for agv_id, agv in agvs.items():
        jev_start = result.jev_plans[agv_id].path[0]
        baseline_start = result.baseline_plans[agv_id].path[0]
        assert jev_start == agv.cell
        assert baseline_start == agv.cell
