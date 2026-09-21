from typesafe_sdk import Noul, Score

from jev_planner.questions import (
    DAMAGE_LEVELS,
    DELAY_LEVELS,
    DIMENSIONS,
    PEOPLE_LEVELS,
    build_questions,
    question_id,
)
from jev_planner.world import load_world_file


def world():
    return load_world_file("tests/fixtures/mini.yaml")


def test_four_questions_per_zone():
    q = build_questions(world())
    assert len(q) == 2 * 4
    assert set(q) == {question_id(z, d) for z in ("aisle-1", "aisle-2") for d in DIMENSIONS}


def test_score_questions_carry_four_ordered_levels():
    q = build_questions(world())
    for dimension, levels in (
        ("people", PEOPLE_LEVELS),
        ("damage", DAMAGE_LEVELS),
        ("delay", DELAY_LEVELS),
    ):
        question = q[question_id("aisle-1", dimension)]
        assert isinstance(question, Score)
        assert list(question.criteria) == list(levels)
        assert len(levels) == 4


def test_offlimits_is_a_noul():
    assert isinstance(build_questions(world())[question_id("aisle-1", "offlimits")], Noul)


def test_instructions_name_the_zone_by_a_backticked_state_path():
    q = build_questions(world())
    assert "`zones.aisle-2`" in q[question_id("aisle-2", "people")].instructions


def test_a_question_mentions_no_other_zone():
    q = build_questions(world())
    assert "aisle-2" not in q[question_id("aisle-1", "damage")].instructions
