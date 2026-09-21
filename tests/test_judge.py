import json

import pytest
from typesafe_sdk import Noul, Score, SystemOneResponse

from jev_planner.judge import Judge, Judgment, MissingAnswerError, cache_key

STATE = {"shift": {"clock": "13:00"}, "zones": {}, "agvs": {}}
QUESTIONS = {
    "aisle-1.people": Score(instructions="how exposed", criteria=["a", "b", "c", "d"]),
    "aisle-1.offlimits": Noul(instructions="closed?"),
}


def response(answers: dict, tokens: int = 100) -> SystemOneResponse:
    return SystemOneResponse.model_validate(
        {"model": "jev-1.13.0", "usage": {"input_tokens": tokens}, "answers": answers}
    )


FULL = {
    "aisle-1.people": {
        "type": "score", "score": 2.0, "confidence": 0.9,
        "legend": {0: "a", 1: "b", 2: "c", 3: "d"},
        "probabilities": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
    },
    "aisle-1.offlimits": {"type": "noul", "noul": 0.8},
}


class FakeClient:
    def __init__(self, answers=FULL):
        self.answers = answers
        self.calls = 0

    def system_one(self, state, questions, **kwargs):
        self.calls += 1
        return response(self.answers)


def test_returns_raw_answers_verbatim(tmp_path):
    j = Judge(client=FakeClient(), cache_path=tmp_path / "c.json").judge(STATE, QUESTIONS)
    assert j.model == "jev-1.13.0"
    assert j.input_tokens == 100
    assert j.cached is False
    assert j.answers["aisle-1.people"]["probabilities"] == {"2": 1.0, "0": 0.0, "1": 0.0, "3": 0.0}
    assert j.answers["aisle-1.people"]["confidence"] == 0.9
    assert j.answers["aisle-1.offlimits"]["noul"] == 0.8


def test_request_id_absent_does_not_raise(tmp_path):
    assert Judge(client=FakeClient(), cache_path=tmp_path / "c.json").judge(
        STATE, QUESTIONS
    ).request_id is None


def test_a_missing_answer_is_a_hard_error(tmp_path):
    partial = {"aisle-1.people": FULL["aisle-1.people"]}
    judge = Judge(client=FakeClient(partial), cache_path=tmp_path / "c.json")
    with pytest.raises(MissingAnswerError, match="aisle-1.offlimits"):
        judge.judge(STATE, QUESTIONS)


def test_second_identical_call_is_served_from_cache(tmp_path):
    client = FakeClient()
    judge = Judge(client=client, cache_path=tmp_path / "c.json")
    first = judge.judge(STATE, QUESTIONS)
    second = judge.judge(STATE, QUESTIONS)
    assert client.calls == 1
    assert second.cached is True
    assert second.answers == first.answers


def test_cache_survives_a_new_judge_instance(tmp_path):
    client = FakeClient()
    Judge(client=client, cache_path=tmp_path / "c.json").judge(STATE, QUESTIONS)
    again = Judge(client=FakeClient(), cache_path=tmp_path / "c.json").judge(STATE, QUESTIONS)
    assert again.cached is True


def test_a_changed_state_misses_the_cache(tmp_path):
    client = FakeClient()
    judge = Judge(client=client, cache_path=tmp_path / "c.json")
    judge.judge(STATE, QUESTIONS)
    judge.judge({**STATE, "shift": {"clock": "14:00"}}, QUESTIONS)
    assert client.calls == 2


def test_cache_key_depends_on_the_model(tmp_path):
    assert cache_key(STATE, QUESTIONS, "jev-1.13.0") != cache_key(STATE, QUESTIONS, "jev-latest")
