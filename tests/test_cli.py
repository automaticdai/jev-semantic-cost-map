import json

from jev_costmap.cli import main


def test_dry_run_prints_the_state_and_questions_and_makes_no_call(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["run", "--scenario", "tests/fixtures/mini-scenario.yaml", "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"state", "questions"}
    assert printed["state"]["shift"]["tick"] == 0
    assert len(printed["questions"]) == 8


def test_run_without_an_api_key_fails_with_a_useful_message(capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code = main(["run", "--scenario", "tests/fixtures/mini-scenario.yaml", "--out", str(tmp_path)])
    assert code == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_unknown_command_is_an_error():
    assert main(["nonsense"]) != 0


def test_explain_on_an_unknown_zone_fails_cleanly(capsys):
    code = main(["explain", "runs/example-night-shift", "--zone", "nowhere"])
    assert code == 2
    err = capsys.readouterr().err
    assert "nowhere" in err
