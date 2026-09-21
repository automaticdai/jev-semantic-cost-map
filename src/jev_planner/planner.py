"""Search over (cell, timestep) on a semantic cost layer.

Everything here is ordinary geometry. The judgments already happened; this
module only reads the numbers they produced.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .costs import CostLayer
from .world import Cell, World

WAIT_COST = 0.5
HORIZON = 400


@dataclass(frozen=True)
class Plan:
    agv: str
    path: tuple[Cell, ...]
    cost: float
    deferred: bool


class Reservations:
    """Time-indexed holds left by already-planned AGVs."""

    def __init__(self, horizon: int = HORIZON) -> None:
        self.horizon = horizon
        self._vertex: set[tuple[Cell, int]] = set()
        self._moves: set[tuple[Cell, Cell, int]] = set()
        self._parked: list[tuple[Cell, int]] = []

    def add(self, path: Sequence[Cell]) -> None:
        for t, cell in enumerate(path):
            self._vertex.add((cell, t))
        for t in range(1, len(path)):
            self._moves.add((path[t - 1], path[t], t))
        self._parked.append((path[-1], len(path) - 1))

    def vertex(self, cell: Cell, t: int) -> bool:
        if (cell, t) in self._vertex:
            return True
        return any(cell == parked and t >= arrival for parked, arrival in self._parked)

    def edge(self, frm: Cell, to: Cell, t: int) -> bool:
        """True if someone else moves `to` -> `frm` arriving at the same timestep."""
        return (to, frm, t) in self._moves


def _passable(world: World, layer: CostLayer, cell: Cell) -> bool:
    if not world.in_bounds(cell):
        return False
    x, y = cell
    return bool(np.isfinite(layer.grid[y, x]))


def _neighbours(cell: Cell) -> tuple[Cell, ...]:
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y))


def plan_single(
    world: World,
    layer: CostLayer,
    agv: str,
    start: Cell,
    goal: Cell,
    reservations: Reservations | None = None,
    horizon: int = HORIZON,
) -> Plan | None:
    if reservations is None:
        reservations = Reservations(horizon)
    if not _passable(world, layer, goal):
        return None

    def h(cell: Cell) -> float:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    counter = itertools.count()
    open_: list[tuple[float, float, int, Cell, int]] = [(h(start), 0.0, next(counter), start, 0)]
    best: dict[tuple[Cell, int], float] = {(start, 0): 0.0}
    parent: dict[tuple[Cell, int], tuple[Cell, int]] = {}

    while open_:
        _, g, _, cell, t = heapq.heappop(open_)
        if g > best.get((cell, t), math.inf):
            continue
        if cell == goal:
            return Plan(agv=agv, path=_reconstruct(parent, cell, t), cost=g, deferred=False)
        if t >= horizon:
            continue
        for nxt in _neighbours(cell):
            if not _passable(world, layer, nxt):
                continue
            if reservations.vertex(nxt, t + 1):
                continue
            if nxt != cell and reservations.edge(cell, nxt, t + 1):
                continue
            step = WAIT_COST if nxt == cell else float(layer.grid[nxt[1], nxt[0]])
            tentative = g + step
            if tentative < best.get((nxt, t + 1), math.inf):
                best[(nxt, t + 1)] = tentative
                parent[(nxt, t + 1)] = (cell, t)
                heapq.heappush(open_, (tentative + h(nxt), tentative, next(counter), nxt, t + 1))

    return None


def _reconstruct(
    parent: Mapping[tuple[Cell, int], tuple[Cell, int]], cell: Cell, t: int
) -> tuple[Cell, ...]:
    path = [cell]
    node = (cell, t)
    while node in parent:
        node = parent[node]
        path.append(node[0])
    path.reverse()
    return tuple(path)


@dataclass(frozen=True)
class PlanRequest:
    agv: str
    priority: int
    start: Cell
    goal: Cell


def plan_all(
    world: World,
    layer: CostLayer,
    requests: Sequence[PlanRequest],
    horizon: int = HORIZON,
) -> dict[str, Plan]:
    """Plan in priority order, each AGV reserving space-time for the next.

    This is prioritized planning, which is incomplete: a low-priority AGV can be
    starved by higher-priority traffic. Starvation surfaces as a deferral rather
    than an exception, and the report lists deferrals per tick.
    """
    reservations = Reservations(horizon)
    plans: dict[str, Plan] = {}
    for request in sorted(requests, key=lambda r: r.priority):
        plan = plan_single(
            world, layer, request.agv, request.start, request.goal, reservations, horizon
        )
        if plan is None:
            plans[request.agv] = Plan(
                agv=request.agv, path=(request.start,), cost=0.0, deferred=True
            )
            continue
        reservations.add(plan.path)
        plans[request.agv] = plan
    return plans
