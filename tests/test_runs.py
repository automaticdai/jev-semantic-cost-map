import datetime as dt

import numpy as np
import pytest

from jev_costmap.runs import RunDir


def test_create_names_the_directory_by_time_and_scenario(tmp_path):
    run = RunDir.create(tmp_path, "night-shift", when=dt.datetime(2026, 9, 21, 14, 2))
    assert run.root.name == "2026-09-21T14-02-night-shift"
    assert run.root.is_dir()


def test_tick_files_round_trip(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    run.write_tick("answers", 7, {"aisle-1.people": {"score": 2.0}})
    assert (run.root / "answers" / "t07.json").exists()
    assert run.read_tick("answers", 7)["aisle-1.people"]["score"] == 2.0


def test_ticks_are_discovered_in_order(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    for tick in (2, 0, 11, 1, 100, 9):
        run.write_tick("states", tick, {})
    assert run.ticks == [0, 1, 2, 9, 11, 100]


def test_manifest_round_trips(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    run.write_manifest({"model": "jev-1.13.0", "input_tokens": 1821})
    assert RunDir.open(run.root).read_manifest()["model"] == "jev-1.13.0"


def test_an_unknown_kind_is_rejected(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    with pytest.raises(ValueError, match="unknown kind"):
        run.write_tick("nonsense", 0, {})


def test_unserializable_values_raise_at_write_site(tmp_path):
    """Unserializable values must raise TypeError when written, not be silently
    converted to strings. A run artifact containing "1.0" (string) instead of
    1.0 (float) is worse than a crash — the corruption would go unnoticed until
    downstream code tried arithmetic, long after the artifact was written."""
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    with pytest.raises(TypeError):
        run.write_tick("costs", 0, {"zone_1": np.float32(1.0)})


def test_open_rejects_a_directory_with_no_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunDir.open(tmp_path / "nope").read_manifest()
