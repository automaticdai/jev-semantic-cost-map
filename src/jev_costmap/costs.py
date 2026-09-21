"""Turn typed judgments into numbers the planner can search over.

The policy lives here, in code, and is deliberately simple: three graded
dimensions become a weighted multiplier, and one yes/no judgment is a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .world import World

BLOCK_THRESHOLD = 0.5
BASE_COST = 1.0

GRADED = ("people", "damage", "delay")


@dataclass(frozen=True)
class Weights:
    people: float
    damage: float
    delay: float

    def of(self, dimension: str) -> float:
        return getattr(self, dimension)


DEFAULT_WEIGHTS = Weights(people=6.0, damage=3.0, delay=1.5)


def load_weights(path: str | Path) -> Weights:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return Weights(
        people=float(data["people"]),
        damage=float(data["damage"]),
        delay=float(data["delay"]),
    )


@dataclass(frozen=True)
class ZoneCost:
    zone: str
    multiplier: float
    blocked: bool
    detail: Mapping[str, float]


@dataclass(frozen=True)
class CostLayer:
    grid: np.ndarray
    zones: Mapping[str, ZoneCost]
    source: str


def _normalised(answer: Mapping) -> float:
    levels = len(answer["legend"])
    return float(answer["score"]) / (levels - 1)


def fuse(world: World, answers: Mapping[str, dict], weights: Weights) -> dict[str, ZoneCost]:
    costs: dict[str, ZoneCost] = {}
    for zone_id in world.zones:
        detail = {d: _normalised(answers[f"{zone_id}.{d}"]) for d in GRADED}
        offlimits = float(answers[f"{zone_id}.offlimits"]["noul"])
        detail["offlimits"] = offlimits
        multiplier = 1.0 + sum(weights.of(d) * detail[d] for d in GRADED)
        costs[zone_id] = ZoneCost(
            zone=zone_id,
            multiplier=multiplier,
            blocked=offlimits >= BLOCK_THRESHOLD,
            detail=detail,
        )
    return costs


def rasterize(world: World, zone_costs: Mapping[str, ZoneCost], source: str) -> CostLayer:
    grid = np.full((world.height, world.width), BASE_COST, dtype=np.float32)
    for zone_id, cost in zone_costs.items():
        zone = world.zones[zone_id]
        window = (slice(zone.y, zone.y + zone.h), slice(zone.x, zone.x + zone.w))
        grid[window] = np.inf if cost.blocked else BASE_COST * cost.multiplier
    grid[world.obstacles] = np.inf
    return CostLayer(grid=grid, zones=dict(zone_costs), source=source)
