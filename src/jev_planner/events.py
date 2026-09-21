"""The scripted shift: events that change what zones mean, tick by tick."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .world import Cell, World, load_world

CLEAR = "clear"
AVOID = "avoid"
BLOCKED = "blocked"
TRUTH_LABELS = (CLEAR, AVOID, BLOCKED)


@dataclass(frozen=True)
class Event:
    zone: str
    note: str


@dataclass(frozen=True)
class AgvState:
    id: str
    cell: Cell
    goal_station: str
    committed_route: tuple[str, ...]
    route_tick: int


@dataclass(frozen=True)
class WorldState:
    tick: int
    clock: str
    scenario: str
    notes: Mapping[str, tuple[str, ...]]
    agvs: Mapping[str, AgvState]
    truth: Mapping[str, str]


@dataclass(frozen=True)
class Scenario:
    name: str
    world: World
    cells_per_tick: int
    clocks: tuple[str, ...]
    events: tuple[tuple[Event, ...], ...]
    truth: tuple[Mapping[str, str], ...]

    @property
    def tick_count(self) -> int:
        return len(self.clocks)


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    with open(path) as fh:
        data = yaml.safe_load(fh)

    world_ref = data["world"]
    if isinstance(world_ref, str):
        with open(path.parent / world_ref) as fh:
            world = load_world(yaml.safe_load(fh))
    else:
        world = load_world(world_ref)

    clocks: list[str] = []
    events: list[tuple[Event, ...]] = []
    truth: list[Mapping[str, str]] = []
    for index, tick in enumerate(data["ticks"]):
        clocks.append(str(tick["clock"]))
        tick_events = tuple(Event(e["zone"], e["note"]) for e in tick.get("events", []))
        for event in tick_events:
            if event.zone not in world.zones:
                raise ValueError(f"tick {index} references unknown zone {event.zone}")
        events.append(tick_events)

        labels = dict.fromkeys(world.zones, CLEAR)
        for zone_id, label in tick.get("truth", {}).items():
            if zone_id not in world.zones:
                raise ValueError(f"tick {index} truth references unknown zone {zone_id}")
            if label not in TRUTH_LABELS:
                raise ValueError(f"tick {index} truth label {label!r} is not one of {TRUTH_LABELS}")
            labels[zone_id] = label
        truth.append(labels)

    return Scenario(
        name=data["name"],
        world=world,
        cells_per_tick=int(data.get("cells_per_tick", 4)),
        clocks=tuple(clocks),
        events=tuple(events),
        truth=tuple(truth),
    )


def initial_agvs(world: World) -> dict[str, AgvState]:
    return {
        spec.id: AgvState(
            id=spec.id,
            cell=world.stations[spec.start].cell,
            goal_station=spec.tasks[0],
            committed_route=(),
            route_tick=-1,
        )
        for spec in world.agvs
    }


def state_at(scenario: Scenario, tick: int, agvs: Mapping[str, AgvState]) -> WorldState:
    notes: dict[str, list[str]] = {zone_id: [] for zone_id in scenario.world.zones}
    for earlier in range(tick + 1):
        for event in scenario.events[earlier]:
            notes[event.zone].append(event.note)

    return WorldState(
        tick=tick,
        clock=scenario.clocks[tick],
        scenario=scenario.name,
        notes={zone_id: tuple(items) for zone_id, items in notes.items()},
        agvs=dict(agvs),
        truth=dict(scenario.truth[tick]),
    )
