import datetime as dt

from jev_planner.baseline import DEFAULT_RULES
from jev_planner.costs import DEFAULT_WEIGHTS
from jev_planner.events import load_scenario
from jev_planner.judge import Judge
from jev_planner.render import layers_from_run, render_run, render_tick
from jev_planner.runs import RunDir
from jev_planner.shift import run_shift
from tests.test_shift import CalmClient


def recorded(tmp_path):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    return s, run


def test_layers_are_rebuilt_from_stored_costs(tmp_path):
    s, run = recorded(tmp_path)
    jev, baseline = layers_from_run(s.world, run, 1)
    assert jev.source == "jev"
    assert baseline.source == "baseline"
    assert jev.grid.shape == (s.world.height, s.world.width)


def test_a_frame_is_written_per_tick(tmp_path):
    s, run = recorded(tmp_path)
    frame = render_tick(s, run, 1)
    assert frame.exists()
    assert frame.stat().st_size > 5000


def test_render_run_produces_a_gif_and_every_frame(tmp_path):
    s, run = recorded(tmp_path)
    gif = render_run(s, run)
    assert gif.exists()
    assert gif.stat().st_size > 5000
    assert len(list(run.frames_dir.glob("t*.png"))) == 3


def test_replay_needs_no_api_key(tmp_path, monkeypatch, capsys):
    from jev_planner.cli import main

    s, run = recorded(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["replay", str(run.root)]) == 0
    assert run.gif_path.exists()
