import datetime as dt

from jev_planner.baseline import DEFAULT_RULES
from jev_planner.costs import DEFAULT_WEIGHTS, ZoneCost
from jev_planner.events import load_scenario
from jev_planner.judge import Judge
from jev_planner.planner import Plan
from jev_planner.report import disagree_text, explain_text, score_run, score_tick, write_report
from jev_planner.runs import RunDir
from jev_planner.shift import run_shift
from jev_planner.world import load_world_file
from tests.test_shift import CalmClient


def world():
    return load_world_file("tests/fixtures/mini.yaml")


def costs(blocked=()):
    return {z: ZoneCost(z, 1.0, z in blocked, {}) for z in ("aisle-1", "aisle-2")}


def test_entering_a_blocked_zone_is_one_violation_per_agv():
    plans = {"agv-1": Plan("agv-1", ((0, 5), (0, 4), (0, 2), (1, 2)), 3.0, False)}
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(), plans)
    assert s.blocked_violations == 1
    assert s.path_cells == 3


def test_the_zone_the_agv_started_in_does_not_count_as_an_entry():
    # agv-1 begins inside aisle-1 and drives out of it. It did not choose to be
    # there, so leaving is not a violation.
    plans = {"agv-1": Plan("agv-1", ((0, 1), (1, 1), (1, 4)), 2.0, False)}
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(), plans)
    assert s.blocked_violations == 0
    assert s.path_cells == 2


def test_avoid_is_counted_separately():
    plans = {"agv-1": Plan("agv-1", ((0, 5), (0, 4), (0, 2)), 2.0, False)}
    s = score_tick(world(), {"aisle-1": "avoid", "aisle-2": "clear"}, costs(), plans)
    assert s.avoid_traversals == 1
    assert s.blocked_violations == 0


def test_blocking_a_clear_zone_is_a_false_block():
    s = score_tick(world(), {"aisle-1": "clear", "aisle-2": "clear"}, costs(blocked=("aisle-1",)), {})
    assert s.false_blocks == 1


def test_blocking_a_genuinely_blocked_zone_is_not_a_false_block():
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(blocked=("aisle-1",)), {})
    assert s.false_blocks == 0


def test_a_deferred_agv_contributes_a_deferral_and_nothing_else():
    plans = {"agv-1": Plan("agv-1", ((0, 1),), 0.0, True)}
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(), plans)
    assert (s.deferrals, s.path_cells, s.blocked_violations) == (1, 0, 0)


def recorded(tmp_path):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    return s, run


def test_score_run_covers_both_sources_and_every_tick(tmp_path):
    s, run = recorded(tmp_path)
    scores = score_run(s, run)
    assert set(scores) == {"jev", "baseline"}
    assert len(scores["jev"]) == 3


def test_the_report_names_both_models_and_the_totals(tmp_path):
    s, run = recorded(tmp_path)
    text = write_report(s, run).read_text()
    assert "jev" in text and "baseline" in text
    assert "blocked violations" in text.lower()


def test_explain_shows_every_probability_and_the_resulting_multiplier(tmp_path):
    s, run = recorded(tmp_path)
    text = explain_text(s, run, "aisle-1", 1)
    assert "people" in text and "damage" in text and "delay" in text
    assert "offlimits" in text
    assert "multiplier" in text
    assert "confidence" in text


def test_explain_rejects_an_unknown_zone(tmp_path):
    s, run = recorded(tmp_path)
    try:
        explain_text(s, run, "nowhere", 1)
    except KeyError:
        return
    raise AssertionError("expected a KeyError for an unknown zone")


def test_disagree_reports_the_spill_the_baseline_will_not_release(tmp_path):
    """At t=2 the mini scenario's spill has been mopped and the truth label is
    back to clear. The calm client leaves aisle-1 open; the baseline still blocks
    on the word. The row must appear and must name the winner."""
    s, run = recorded(tmp_path)
    text = disagree_text(s, run)
    assert "aisle-1" in text
    assert "jev right" in text


def test_a_difference_of_degree_is_not_scored_as_a_win(tmp_path):
    from jev_planner.report import _verdict

    assert _verdict(False, False, "clear") == "degree only"
    assert _verdict(False, False, "avoid") == "degree only"
    assert _verdict(False, False, "blocked") == "both wrong"
    assert _verdict(True, False, "blocked") == "jev right"
    assert _verdict(False, True, "clear") == "jev right"
    assert _verdict(True, False, "clear") == "baseline right"
