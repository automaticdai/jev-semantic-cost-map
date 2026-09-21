"""Scoring against the hidden labels, and the three things you read afterwards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .costs import ZoneCost
from .events import Scenario
from .planner import Plan
from .runs import RunDir
from .world import Cell, World

SOURCES = ("jev", "baseline")


@dataclass(frozen=True)
class TickScore:
    tick: int
    source: str
    blocked_violations: int
    avoid_traversals: int
    false_blocks: int
    path_cells: int
    deferrals: int


def _zones_entered(world: World, path: Sequence[Cell]) -> set[str]:
    """Zones the AGV moved into this tick.

    The zone it was already standing in is excluded: it did not choose to enter
    that one, so charging it a violation would punish an AGV for where the last
    tick left it.
    """
    start = world.zone_at(path[0])
    return {z for z in (world.zone_at(cell) for cell in path[1:]) if z and z != start}


def score_tick(
    world: World,
    truth: Mapping[str, str],
    zone_costs: Mapping[str, ZoneCost],
    plans: Mapping[str, Plan],
    tick: int = 0,
    source: str = "jev",
) -> TickScore:
    blocked = avoid = cells = deferrals = 0
    for plan in plans.values():
        if plan.deferred:
            deferrals += 1
            continue
        cells += len(plan.path) - 1
        for zone in _zones_entered(world, plan.path):
            if truth.get(zone) == "blocked":
                blocked += 1
            elif truth.get(zone) == "avoid":
                avoid += 1
    false_blocks = sum(
        1 for zone, cost in zone_costs.items() if cost.blocked and truth.get(zone) == "clear"
    )
    return TickScore(tick, source, blocked, avoid, false_blocks, cells, deferrals)


def _plans_of(run: RunDir, tick: int, source: str) -> dict[str, Plan]:
    stored = run.read_tick("plans", tick)[source]
    return {
        agv: Plan(agv, tuple(tuple(c) for c in p["path"]), p["cost"], p["deferred"])
        for agv, p in stored.items()
    }


def _costs_of(run: RunDir, tick: int, source: str) -> dict[str, ZoneCost]:
    stored = run.read_tick("costs", tick)[source]
    return {
        zone: ZoneCost(zone, float(e["multiplier"]), bool(e["blocked"]), e["detail"])
        for zone, e in stored.items()
    }


def score_run(scenario: Scenario, run: RunDir) -> dict[str, list[TickScore]]:
    world = scenario.world
    out: dict[str, list[TickScore]] = {source: [] for source in SOURCES}
    for tick in run.ticks:
        truth = run.read_tick("truth", tick)
        for source in SOURCES:
            out[source].append(
                score_tick(world, truth, _costs_of(run, tick, source),
                           _plans_of(run, tick, source), tick, source)
            )
    return out


def _totals(scores: Sequence[TickScore]) -> dict[str, int]:
    return {
        "blocked violations": sum(s.blocked_violations for s in scores),
        "avoid traversals": sum(s.avoid_traversals for s in scores),
        "false blocks": sum(s.false_blocks for s in scores),
        "path cells": sum(s.path_cells for s in scores),
        "deferrals": sum(s.deferrals for s in scores),
    }


def write_report(scenario: Scenario, run: RunDir) -> Path:
    manifest = run.read_manifest()
    scores = score_run(scenario, run)
    jev, base = _totals(scores["jev"]), _totals(scores["baseline"])

    lines = [
        f"# {scenario.name}",
        "",
        f"Model `{manifest['model']}`, {manifest['input_tokens']} input tokens, "
        f"about ${manifest['estimated_usd']}.",
        "",
        "## Scored against the hidden labels",
        "",
        "| Measure | jev | baseline |",
        "| --- | ---: | ---: |",
    ]
    for key in jev:
        lines.append(f"| {key} | {jev[key]} | {base[key]}  |")

    lines += ["", "## Per tick", "",
              "| tick | clock | jev blocked | base blocked | jev cells | base cells | deferred |",
              "| ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for entry, j, b in zip(manifest["ticks"], scores["jev"], scores["baseline"]):
        lines.append(
            f"| {entry['tick']} | {entry['clock']} | {j.blocked_violations} | "
            f"{b.blocked_violations} | {j.path_cells} | {b.path_cells} | "
            f"{', '.join(entry['deferred']) or '-'} |"
        )

    lines += ["", "## Where they disagreed", "", disagree_text(scenario, run)]
    run.report_path.write_text("\n".join(lines) + "\n")
    return run.report_path


def explain_text(scenario: Scenario, run: RunDir, zone: str, tick: int) -> str:
    if zone not in scenario.world.zones:
        raise KeyError(f"unknown zone {zone!r}")
    answers = run.read_tick("answers", tick)
    jev = _costs_of(run, tick, "jev")[zone]
    base = _costs_of(run, tick, "baseline")[zone]
    state = run.read_tick("states", tick)

    lines = [f"{zone} at t={tick} ({scenario.clocks[tick]})", ""]
    for note in state["zones"][zone]["notes"] or ["(no notes)"]:
        lines.append(f"  note: {note}")
    lines.append("")
    for dimension in ("people", "damage", "delay"):
        answer = answers[f"{zone}.{dimension}"]
        levels = len(answer["legend"])
        spread = "  ".join(
            f"[{i}] {answer['probabilities'][str(i)]:.2f}" for i in range(levels)
        )
        lines.append(
            f"  {dimension:<10} {answer['score']:.2f}/{levels - 1}  "
            f"confidence {answer['confidence']:.2f}   {spread}"
        )
        for index in range(levels):
            lines.append(f"      [{index}] {answer['legend'][str(index)]}")
    lines.append(f"  {'offlimits':<10} P(closed) = {answers[f'{zone}.offlimits']['noul']:.2f}")
    lines.append("")
    lines.append(f"  jev      multiplier {jev.multiplier:.2f}  blocked={jev.blocked}")
    fired = ", ".join(base.detail) or "no rule fired"
    lines.append(f"  baseline multiplier {base.multiplier:.2f}  blocked={base.blocked}  ({fired})")
    lines.append(f"  truth    {run.read_tick('truth', tick)[zone]}")
    return "\n".join(lines)


def _verdict(jev_blocked: bool, base_blocked: bool, truth: str) -> str:
    def right(blocked: bool) -> bool:
        return blocked if truth == "blocked" else not blocked

    if not jev_blocked and not base_blocked:
        # Neither called it closed, so this row is a difference of degree.
        # Calling that a win for either model would be reading too much into it.
        return "both wrong" if truth == "blocked" else "degree only"
    if right(jev_blocked) and not right(base_blocked):
        return "jev right"
    if right(base_blocked) and not right(jev_blocked):
        return "baseline right"
    return "both right" if right(jev_blocked) else "both wrong"


def disagree_text(scenario: Scenario, run: RunDir) -> str:
    rows = ["| tick | zone | jev | baseline | truth | verdict |",
            "| ---: | --- | --- | --- | --- | --- |"]
    for tick in run.ticks:
        truth = run.read_tick("truth", tick)
        jev_costs = _costs_of(run, tick, "jev")
        base_costs = _costs_of(run, tick, "baseline")
        for zone in scenario.world.zones:
            j, b = jev_costs[zone], base_costs[zone]
            differs = j.blocked != b.blocked or max(j.multiplier, b.multiplier) > 1.5 * min(
                j.multiplier, b.multiplier
            )
            if not differs:
                continue
            rows.append(
                f"| {tick} | {zone} | "
                f"{'blocked' if j.blocked else f'x{j.multiplier:.1f}'} | "
                f"{'blocked' if b.blocked else f'x{b.multiplier:.1f}'} | "
                f"{truth[zone]} | {_verdict(j.blocked, b.blocked, truth[zone])} |"
            )
    if len(rows) == 2:
        return "The two models agreed everywhere."
    return "\n".join(rows)
