import datetime as dt

import pytest

from jev_costmap.baseline import DEFAULT_RULES
from jev_costmap.costs import DEFAULT_WEIGHTS
from jev_costmap.events import load_scenario
from jev_costmap.judge import Judge
from jev_costmap.render import layers_from_run, paths_from_run, render_run, render_tick
from jev_costmap.runs import RunDir
from jev_costmap.shift import run_shift
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


def test_render_tick_pairs_each_panel_with_its_own_layer_and_paths(tmp_path, monkeypatch):
    """The left panel must show Jev's cost layer with Jev's paths and the right
    panel the baseline's; crossing them would be invisible in a file-size check
    and would misrepresent every frame of the demo."""
    s, run = recorded(tmp_path)
    calls = []

    def recorder(*args, **kwargs):
        calls.append(args)

    monkeypatch.setattr("jev_costmap.render._panel", recorder)
    # With _panel stubbed out, axes[0] never gets a labeled artist, so
    # render_tick's own axes[0].legend() call raises its usual, harmless
    # UserWarning; expect it explicitly rather than let it leak into the
    # suite's warning summary.
    with pytest.warns(UserWarning, match="No artists with labels"):
        render_tick(s, run, 1)

    assert len(calls) == 2
    expected_paths = paths_from_run(run, 1)

    _, _, jev_layer, jev_paths, jev_title = calls[0]
    _, _, baseline_layer, baseline_paths, baseline_title = calls[1]

    assert jev_layer.source == "jev"
    assert jev_paths == expected_paths["jev"]
    assert jev_title == "Jev semantic cost layer"

    assert baseline_layer.source == "baseline"
    assert baseline_paths == expected_paths["baseline"]
    assert baseline_title == "Keyword rule baseline"


def test_render_run_produces_a_gif_and_every_frame(tmp_path):
    s, run = recorded(tmp_path)
    gif = render_run(s, run)
    assert gif.exists()
    assert gif.stat().st_size > 5000
    assert len(list(run.frames_dir.glob("t*.png"))) == 3


def test_replay_needs_no_api_key(tmp_path, monkeypatch, capsys):
    from jev_costmap.cli import main

    s, run = recorded(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["replay", str(run.root)]) == 0
    assert run.gif_path.exists()
