"""A keyword cost model, planned alongside Jev so the demo can be checked.

It is not a strawman: it catches every case someone thought to add a word for.
What it cannot do is read a correction, a time, or a spatial qualifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from .costs import CostLayer, ZoneCost, rasterize
from .events import WorldState
from .world import World


@dataclass(frozen=True)
class Rule:
    pattern: str
    multiplier: float | None
    blocked: bool


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("spill|wet floor|leak|leaking", None, True),
    Rule("closed|do not enter|no entry|out of service", None, True),
    Rule("picker|pickers|staff|operator|person|people|engineer", 3.0, False),
    Rule("fragile|glass|glassware|breakable", 2.0, False),
    Rule("obstruction|blocked aisle|pallet left", 2.0, False),
)


def load_rules(path: str | Path) -> tuple[Rule, ...]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return tuple(
        Rule(
            pattern=item["pattern"],
            multiplier=float(item["multiplier"]) if "multiplier" in item else None,
            blocked=bool(item.get("blocked", False)),
        )
        for item in data
    )


def baseline_costs(
    world: World, ws: WorldState, rules: Sequence[Rule] = DEFAULT_RULES
) -> dict[str, ZoneCost]:
    costs: dict[str, ZoneCost] = {}
    for zone_id, zone in world.zones.items():
        text = "\n".join((zone.description, *ws.notes.get(zone_id, ()))).lower()
        blocked = False
        multiplier = 1.0
        detail: dict[str, float] = {}
        for rule in rules:
            if not re.search(rule.pattern, text):
                continue
            if rule.blocked:
                blocked = True
                detail[rule.pattern] = float("inf")
            elif rule.multiplier is not None:
                multiplier *= rule.multiplier
                detail[rule.pattern] = rule.multiplier
        costs[zone_id] = ZoneCost(
            zone=zone_id, multiplier=multiplier, blocked=blocked, detail=detail
        )
    return costs


def baseline_layer(
    world: World, ws: WorldState, rules: Sequence[Rule] = DEFAULT_RULES
) -> CostLayer:
    return rasterize(world, baseline_costs(world, ws, rules), source="baseline")
