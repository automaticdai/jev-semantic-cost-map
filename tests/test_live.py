import os

import pytest

from jev_planner.events import initial_agvs, load_scenario, state_at
from jev_planner.judge import Judge
from jev_planner.questions import build_questions
from jev_planner.state import build_state

pytestmark = pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"), reason="no TYPESAFE_API_KEY"
)


@pytest.mark.live
def test_one_real_request_answers_every_question(tmp_path):
    s = load_scenario("config/scenarios/night-shift.yaml")
    ws = state_at(s, 3, initial_agvs(s.world))
    questions = build_questions(s.world)

    judgment = Judge(cache_path=tmp_path / "c.json").judge(build_state(s.world, ws), questions)

    assert set(judgment.answers) == set(questions)
    assert judgment.model.startswith("jev-")
    assert judgment.input_tokens > 0
    for qid, answer in judgment.answers.items():
        if qid.endswith(".offlimits"):
            assert 0.0 <= answer["noul"] <= 1.0
        else:
            assert 0.0 <= answer["score"] <= 3.0
            assert 0.0 <= answer["confidence"] <= 1.0
            assert len(answer["probabilities"]) == 4
