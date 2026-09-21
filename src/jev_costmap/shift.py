"""One tick: judge the scene, price the map, plan every AGV, advance."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence

from typesafe_sdk import TypeSafeAPIError

from .baseline import Rule, baseline_costs
from .costs import CostLayer, Weights, ZoneCost, fuse, rasterize
from .events import AgvState, Scenario, WorldState, initial_agvs, state_at
from .judge import Judge, Judgment
from .planner import Plan, PlanRequest, plan_all
from .questions import build_questions
from .runs import RunDir
from .state import build_state
from .world import Cell, World


# Jev's published price, from the pricing table at https://console.typesafe.ai/,
# as of 2026-09: $0.042 per million input tokens.
JEV_PRICE_PER_MTOK_USD = 0.042


class ShiftAborted(RuntimeError):
    """A tick failed for good. Artifacts already written stay on disk."""


def _git_sha() -> str | None:
    """The current HEAD sha, or None if git is absent or this isn't a repo.

    Recorded in the manifest per the spec's run-directory section, so a run
    can always be traced back to the code that produced it. Never lets a
    missing git binary or a non-repo working directory crash a run.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class TickResult:
    tick: int
    state: WorldState
    judgment: Judgment
    jev_costs: Mapping[str, ZoneCost]
    baseline_costs: Mapping[str, ZoneCost]
    jev_layer: CostLayer
    baseline_layer: CostLayer
    jev_plans: Mapping[str, Plan]
    baseline_plans: Mapping[str, Plan]


def _route_zones(world: World, path: Sequence[Cell]) -> tuple[str, ...]:
    zones: list[str] = []
    for cell in path:
        zone = world.zone_at(cell)
        if zone and (not zones or zones[-1] != zone):
            zones.append(zone)
    return tuple(zones)


def _requests(scenario: Scenario, agvs: Mapping[str, AgvState]) -> list[PlanRequest]:
    world = scenario.world
    priority = {spec.id: spec.priority for spec in world.agvs}
    return [
        PlanRequest(
            agv=agv.id,
            priority=priority[agv.id],
            start=agv.cell,
            goal=world.stations[agv.goal_station].cell,
        )
        for agv in agvs.values()
    ]


def advance(
    scenario: Scenario, agvs: Mapping[str, AgvState], plans: Mapping[str, Plan], tick: int
) -> dict[str, AgvState]:
    world = scenario.world
    tasks = {spec.id: spec.tasks for spec in world.agvs}
    moved: dict[str, AgvState] = {}
    for agv_id, agv in agvs.items():
        plan = plans[agv_id]
        if plan.deferred:
            moved[agv_id] = AgvState(agv_id, agv.cell, agv.goal_station, (), tick)
            continue
        steps = min(scenario.cells_per_tick, len(plan.path) - 1)
        cell = plan.path[steps]
        remaining = plan.path[steps:]
        goal_station = agv.goal_station
        if cell == world.stations[goal_station].cell:
            queue = tasks[agv_id]
            goal_station = queue[(queue.index(goal_station) + 1) % len(queue)]
        moved[agv_id] = AgvState(
            id=agv_id,
            cell=cell,
            goal_station=goal_station,
            committed_route=_route_zones(world, remaining),
            route_tick=tick,
        )
    return moved


def run_tick(
    scenario: Scenario,
    judge: Judge,
    agvs: Mapping[str, AgvState],
    tick: int,
    weights: Weights,
    rules: Sequence[Rule],
) -> TickResult:
    world = scenario.world
    ws = state_at(scenario, tick, agvs)
    questions = build_questions(world)
    judgment = judge.judge(build_state(world, ws), questions)

    jev_zone_costs = fuse(world, judgment.answers, weights)
    base_zone_costs = baseline_costs(world, ws, rules)
    jev_layer = rasterize(world, jev_zone_costs, source="jev")
    base_layer = rasterize(world, base_zone_costs, source="baseline")

    requests = _requests(scenario, agvs)
    return TickResult(
        tick=tick,
        state=ws,
        judgment=judgment,
        jev_costs=jev_zone_costs,
        baseline_costs=base_zone_costs,
        jev_layer=jev_layer,
        baseline_layer=base_layer,
        jev_plans=plan_all(world, jev_layer, requests),
        baseline_plans=plan_all(world, base_layer, requests),
    )


def _plans_json(plans: Mapping[str, Plan]) -> dict:
    return {
        agv: {"path": [list(c) for c in plan.path], "cost": plan.cost, "deferred": plan.deferred}
        for agv, plan in plans.items()
    }


def _costs_json(costs: Mapping[str, ZoneCost]) -> dict:
    return {
        zone: {"multiplier": c.multiplier, "blocked": c.blocked, "detail": dict(c.detail)}
        for zone, c in costs.items()
    }


def run_shift(
    scenario: Scenario, judge: Judge, run: RunDir, weights: Weights, rules: Sequence[Rule]
) -> dict:
    world = scenario.world
    agvs = initial_agvs(world)
    tick_entries: list[dict] = []
    total_tokens = 0
    model = ""
    git_sha = _git_sha()

    def finish(failure: dict | None = None) -> dict:
        if failure:
            tick_entries.append(failure)
        manifest = {
            "scenario": scenario.name,
            "scenario_path": str(scenario.path),
            "model": model,
            "git_sha": git_sha,
            "input_tokens": total_tokens,
            "estimated_usd": round(total_tokens * JEV_PRICE_PER_MTOK_USD / 1_000_000, 6),
            "weights": {
                "people": weights.people,
                "damage": weights.damage,
                "delay": weights.delay,
            },
            "rules": [
                {"pattern": r.pattern, "multiplier": r.multiplier, "blocked": r.blocked}
                for r in rules
            ],
            "ticks": tick_entries,
        }
        run.write_manifest(manifest)
        return manifest

    for tick in range(scenario.tick_count):
        try:
            result = run_tick(scenario, judge, agvs, tick, weights, rules)
        except TypeSafeAPIError as error:
            # The SDK has already retried. Keep everything written so far, record
            # the failure, and stop. Falling back to the baseline here would hide
            # exactly what this demo exists to show.
            finish({"tick": tick, "clock": scenario.clocks[tick],
                    "status": "failed", "error": str(error)})
            raise ShiftAborted(f"tick {tick} failed: {error}") from error
        model = result.judgment.model
        total_tokens += 0 if result.judgment.cached else result.judgment.input_tokens

        run.write_tick("states", tick, build_state(world, result.state))
        run.write_tick("questions", tick, {
            qid: q.model_dump(mode="json") for qid, q in build_questions(world).items()
        })
        run.write_tick("answers", tick, result.judgment.answers)
        run.write_tick("costs", tick, {
            "jev": _costs_json(result.jev_costs),
            "baseline": _costs_json(result.baseline_costs),
        })
        run.write_tick("plans", tick, {
            "jev": _plans_json(result.jev_plans),
            "baseline": _plans_json(result.baseline_plans),
        })
        run.write_tick("truth", tick, dict(result.state.truth))

        tick_entries.append({
            "tick": tick,
            "clock": result.state.clock,
            "status": "ok",
            "cached": result.judgment.cached,
            "input_tokens": result.judgment.input_tokens,
            "request_id": result.judgment.request_id,
            "deferred": sorted(a for a, p in result.jev_plans.items() if p.deferred),
        })
        agvs = advance(scenario, agvs, result.jev_plans, tick)

    return finish()
