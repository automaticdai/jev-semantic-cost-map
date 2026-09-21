"""Four judgments per zone: three graded dimensions and one hard gate."""

from __future__ import annotations

from typesafe_sdk import Noul, Score

from .world import World

DIMENSIONS = ("people", "damage", "delay", "offlimits")

PEOPLE_LEVELS = (
    "No one works in this zone and no one is expected in it during this shift.",
    "People pass through occasionally but no one is working here now.",
    "People are working in this zone now, clear of the travel lane.",
    "People are working in or across the travel lane, where an AGV would pass.",
)

DAMAGE_LEVELS = (
    "Nothing here is vulnerable; a clear, rated travel surface.",
    "Goods are present but robust and clear of the lane.",
    "Fragile goods, loose material, or an obstruction sits close to the lane.",
    "The surface or the load is compromised in a way that makes a pass likely to "
    "cause damage or a loss of traction.",
)

DELAY_LEVELS = (
    "Clear; an AGV passes at full speed.",
    "Light traffic or a narrow point; a short slowdown.",
    "Busy; an AGV would likely have to stop and wait.",
    "Effectively impassable without a long wait.",
)


def question_id(zone_id: str, dimension: str) -> str:
    return f"{zone_id}.{dimension}"


def build_questions(world: World) -> dict[str, Score | Noul]:
    questions: dict[str, Score | Noul] = {}
    for zone_id in world.zones:
        path = f"`zones.{zone_id}`"
        questions[question_id(zone_id, "people")] = Score(
            instructions=(
                f"Considering only zone {path}, how exposed would a person be if an "
                "AGV drove through it right now? Read the zone's notes in order; a "
                "later note can cancel an earlier one."
            ),
            criteria=list(PEOPLE_LEVELS),
        )
        questions[question_id(zone_id, "damage")] = Score(
            instructions=(
                f"Considering only zone {path}, how likely is an AGV passing through "
                "it right now to damage goods, the floor surface, or itself? Read the "
                "zone's notes in order; a later note can cancel an earlier one."
            ),
            criteria=list(DAMAGE_LEVELS),
        )
        questions[question_id(zone_id, "delay")] = Score(
            instructions=(
                f"Considering only zone {path}, how much would an AGV be slowed by "
                "congestion or obstruction in it right now? Judge delay alone, "
                "ignoring how safe or damaging the pass would be."
            ),
            criteria=list(DELAY_LEVELS),
        )
        questions[question_id(zone_id, "offlimits")] = Noul(
            instructions=(
                f"Is zone {path} closed to AGV traffic right now by an explicit "
                "instruction, an active incident, or a standing rule? Answer no if "
                "the zone is merely a poor choice rather than closed."
            ),
        )
    return questions
