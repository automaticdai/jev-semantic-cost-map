# jev-semantic-cost-map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI demo in which Jev judges named warehouse zones against a natural-language scene, code turns those judgments into a cost layer, and four AGVs replan over it as a scripted shift unfolds.

**Architecture:** One Jev request per tick carries the whole warehouse as `state` and four questions per zone; `costs.py` fuses the typed answers into a per-zone multiplier and a blocked set, rasterized onto an occupancy grid; a prioritized space-time A\* routes the AGVs. A keyword rule baseline plans the same states, and hidden per-tick ground-truth labels — never sent to the model — let the report score both.

**Tech Stack:** Python 3.12, uv, `typesafe-sdk` 0.7, numpy, matplotlib (+pillow for GIF), pyyaml, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-jev-semantic-cost-map-design.md`

## Global Constraints

- Python 3.12; dependencies managed with `uv`. Package `jev_costmap` under `src/`, console script `jev-costmap`.
- Grid arrays are numpy and indexed `[y, x]`. Cells are `(x, y)` tuples everywhere else. Never mix these.
- Only `judge.py` performs network I/O. Every other module operates on stored data.
- The default pytest suite makes no network calls. The one live test is `@pytest.mark.live` and skips without `TYPESAFE_API_KEY`.
- Ground-truth labels must never appear in the state sent to Jev. `test_state.py` enforces this.
- Raw answers are stored verbatim; no module may discard level probabilities or confidence.
- `NoulAnswer` carries only `noul` — there is **no** confidence field on a Noul. `ScoreAnswer` carries `score`, `confidence`, `legend`, `probabilities`.
- Jev failing never falls back to the baseline.
- Fusion weights, as committed starting values: `people: 6.0`, `damage: 3.0`, `delay: 1.5`.
- Block threshold: `offlimits.noul >= 0.5`.
- Base traversal cost is `1.0` per free cell. Waiting in place costs `0.5`, independent of zone.
- Planning horizon: 400 timesteps.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Verified SDK facts

Confirmed against `typesafe-sdk` 0.7.0 and one live `jev-1.13.0` call before this plan was written:

```python
from typesafe_sdk import TypeSafeClient, Score, Noul
r = TypeSafeClient().system_one(state_dict, {"aisle-3.people": Score(instructions=..., criteria=[...])})
r.model            # "jev-1.13.0"
r.usage.input_tokens
r.scores["aisle-3.people"].score          # float, probability-weighted, 0..len(levels)-1
r.scores["aisle-3.people"].probabilities  # dict[int, float]
r.scores["aisle-3.people"].confidence     # float
r.scores["aisle-3.people"].legend         # dict[int, str]
r.nouls["aisle-3.offlimits"].noul         # float, no confidence attribute
r.answers                                  # dict[str, ScoreAnswer|NoulAnswer|ChoiceAnswer], all ids
r.request_id
```

Question ids containing `.` and `-` are accepted.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | packaging, deps, console script, pytest config |
| `src/jev_costmap/world.py` | `Zone`, `Station`, `AgvSpec`, `World`, `load_world`; rasterization |
| `src/jev_costmap/events.py` | `Event`, `AgvState`, `WorldState`, `Scenario`, `load_scenario`, `state_at` |
| `src/jev_costmap/state.py` | `build_state` — `WorldState` → the JSON dict sent to Jev |
| `src/jev_costmap/questions.py` | level texts, `build_questions` |
| `src/jev_costmap/judge.py` | `Judgment`, `Judge`, `MissingAnswerError`, on-disk cache |
| `src/jev_costmap/costs.py` | `Weights`, `ZoneCost`, `CostLayer`, `fuse`, `rasterize` |
| `src/jev_costmap/baseline.py` | `Rule`, `load_rules`, `baseline_layer` |
| `src/jev_costmap/planner.py` | `Plan`, `Reservations`, `plan_single`, `plan_all` |
| `src/jev_costmap/runs.py` | `RunDir`, manifest read/write |
| `src/jev_costmap/shift.py` | the per-tick orchestration loop (an addition to the spec's module list: the loop does not belong in `cli.py`) |
| `src/jev_costmap/render.py` | frames and GIF |
| `src/jev_costmap/report.py` | scoring, `report.md`, `explain`, `disagree` |
| `src/jev_costmap/cli.py` | argparse subcommands |
| `config/weights.yaml`, `config/rules.yaml` | fusion weights, baseline rules |
| `config/scenarios/night-shift.yaml` | the primary scenario |
| `tests/fixtures/mini.yaml` | a 10x8 two-aisle world used by most tests |
| `tests/fixtures/answers/*.json` | real recorded Jev responses |

---

### Task 1: Project scaffolding and the world model

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/jev_costmap/__init__.py`, `src/jev_costmap/world.py`, `tests/fixtures/mini.yaml`
- Test: `tests/test_world.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Zone(id, name, kind, description, x, y, w, h)`, `Station(id, cell)`, `AgvSpec(id, priority, start, tasks)`, `World(width, height, cell_size_m, zones, stations, agvs, obstacles, zone_grid, zone_ids)`, `load_world(data: dict) -> World`, `World.zone_at(cell) -> str | None`, `World.passable(cell) -> bool`. `obstacles` and `zone_grid` are numpy arrays shaped `(height, width)`.

- [ ] **Step 1: Write the failing test**

`tests/fixtures/mini.yaml`:

```yaml
width: 10
height: 8
cell_size_m: 0.5
walls:
  - {x: 0, y: 3, w: 10, h: 1}
  - {x: 4, y: 3, w: 1, h: 1, clear: true}
zones:
  - id: aisle-1
    name: Aisle 1
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic.
    x: 0
    y: 0
    w: 10
    h: 3
  - id: aisle-2
    name: Aisle 2
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic.
    x: 0
    y: 4
    w: 10
    h: 4
stations:
  - {id: dock, cell: [0, 0]}
  - {id: bay, cell: [9, 7]}
agvs:
  - {id: agv-1, priority: 1, start: dock, tasks: [bay]}
```

`tests/test_world.py`:

```python
import numpy as np
import pytest
import yaml

from jev_costmap.world import load_world


def mini() -> dict:
    with open("tests/fixtures/mini.yaml") as fh:
        return yaml.safe_load(fh)


def test_grid_shape_and_zone_rasterization():
    w = load_world(mini())
    assert w.obstacles.shape == (8, 10)
    assert w.zone_grid.shape == (8, 10)
    assert w.zone_at((3, 1)) == "aisle-1"
    assert w.zone_at((3, 5)) == "aisle-2"
    assert w.zone_at((3, 3)) is None  # the wall row belongs to no zone


def test_walls_block_and_clear_punches_a_doorway():
    w = load_world(mini())
    assert not w.passable((3, 3))
    assert w.passable((4, 3))
    assert w.passable((0, 0))


def test_out_of_bounds_is_not_passable():
    w = load_world(mini())
    assert not w.passable((-1, 0))
    assert not w.passable((10, 0))
    assert not w.passable((0, 8))


def test_stations_and_agvs_are_loaded():
    w = load_world(mini())
    assert w.stations["bay"].cell == (9, 7)
    assert w.agvs[0].id == "agv-1"
    assert w.agvs[0].tasks == ("bay",)


def test_overlapping_zones_are_rejected():
    data = mini()
    data["zones"].append(
        {"id": "aisle-3", "name": "A3", "kind": "travel lane",
         "description": "d", "x": 0, "y": 0, "w": 2, "h": 2}
    )
    with pytest.raises(ValueError, match="overlap"):
        load_world(data)


def test_zone_outside_the_grid_is_rejected():
    data = mini()
    data["zones"][0]["w"] = 99
    with pytest.raises(ValueError, match="outside the grid"):
        load_world(data)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_world.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'jev_costmap'`.

- [ ] **Step 3: Write the scaffolding**

`pyproject.toml`:

```toml
[project]
name = "jev-semantic-cost-map"
version = "0.1.0"
description = "Path planning where Jev judges the scene and code owns the geometry"
requires-python = ">=3.12"
dependencies = [
    "typesafe-sdk>=0.7",
    "numpy>=2.0",
    "matplotlib>=3.9",
    "pillow>=10.0",
    "pyyaml>=6.0",
]

[project.scripts]
jev-costmap = "jev_costmap.cli:main"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jev_costmap"]

[tool.pytest.ini_options]
testpaths = ["tests"]
# `pythonpath` puts the repo root on sys.path so `from tests.helpers import ...`
# resolves under the src layout. `addopts` keeps the live test out of the default
# suite even when TYPESAFE_API_KEY is exported; `-m live` on the command line
# overrides it.
pythonpath = ["."]
addopts = ["-m", "not live"]
markers = ["live: hits the real TypeSafe API; skipped without TYPESAFE_API_KEY"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.jev-cache.json
runs/*
!runs/example-night-shift/
```

`src/jev_costmap/__init__.py`: empty file.

- [ ] **Step 4: Write the world model**

`src/jev_costmap/world.py`:

```python
"""The static warehouse: geometry, named zones, stations and AGV specs.

Grid arrays are numpy, shaped (height, width) and indexed [y, x]. Cells are
(x, y) tuples everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

Cell = tuple[int, int]

NO_ZONE = -1


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    kind: str
    description: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Station:
    id: str
    cell: Cell


@dataclass(frozen=True)
class AgvSpec:
    id: str
    priority: int
    start: str
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class World:
    width: int
    height: int
    cell_size_m: float
    zones: Mapping[str, Zone]
    stations: Mapping[str, Station]
    agvs: tuple[AgvSpec, ...]
    obstacles: np.ndarray
    zone_grid: np.ndarray
    zone_ids: tuple[str, ...]

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, cell: Cell) -> bool:
        if not self.in_bounds(cell):
            return False
        x, y = cell
        return not bool(self.obstacles[y, x])

    def zone_at(self, cell: Cell) -> str | None:
        if not self.in_bounds(cell):
            return None
        x, y = cell
        idx = int(self.zone_grid[y, x])
        return None if idx == NO_ZONE else self.zone_ids[idx]


def load_world(data: dict) -> World:
    width = int(data["width"])
    height = int(data["height"])

    obstacles = np.zeros((height, width), dtype=bool)
    for rect in data.get("walls", []):
        _check_bounds(rect, width, height, "wall")
        value = not rect.get("clear", False)
        obstacles[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"]] = value

    zone_ids: list[str] = []
    zones: dict[str, Zone] = {}
    zone_grid = np.full((height, width), NO_ZONE, dtype=np.int16)
    for spec in data["zones"]:
        zone = Zone(**spec)
        _check_bounds(spec, width, height, f"zone {zone.id}")
        window = zone_grid[zone.y:zone.y + zone.h, zone.x:zone.x + zone.w]
        if (window != NO_ZONE).any():
            raise ValueError(f"zone {zone.id} overlaps another zone")
        window[:] = len(zone_ids)
        zone_ids.append(zone.id)
        zones[zone.id] = zone

    stations = {s["id"]: Station(s["id"], tuple(s["cell"])) for s in data["stations"]}
    agvs = tuple(
        AgvSpec(a["id"], int(a["priority"]), a["start"], tuple(a["tasks"]))
        for a in data["agvs"]
    )
    for agv in agvs:
        for station_id in (agv.start, *agv.tasks):
            if station_id not in stations:
                raise ValueError(f"{agv.id} references unknown station {station_id}")

    return World(
        width=width,
        height=height,
        cell_size_m=float(data.get("cell_size_m", 0.5)),
        zones=zones,
        stations=stations,
        agvs=agvs,
        obstacles=obstacles,
        zone_grid=zone_grid,
        zone_ids=tuple(zone_ids),
    )


def load_world_file(path: str | Path) -> World:
    with open(path) as fh:
        return load_world(yaml.safe_load(fh))


def _check_bounds(rect: dict, width: int, height: int, what: str) -> None:
    if (
        rect["x"] < 0
        or rect["y"] < 0
        or rect["x"] + rect["w"] > width
        or rect["y"] + rect["h"] > height
    ):
        raise ValueError(f"{what} lies outside the grid")
```

- [ ] **Step 5: Run the tests**

```bash
uv sync
uv run pytest tests/test_world.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src tests
git commit -m "$(printf 'feat: warehouse world model and project scaffolding\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Scenario and per-tick world state

**Files:**
- Create: `src/jev_costmap/events.py`
- Test: `tests/test_events.py`
- Modify: `tests/fixtures/mini.yaml` is reused unchanged; add `tests/fixtures/mini-scenario.yaml`

**Interfaces:**
- Consumes: `World`, `load_world`, `AgvSpec` from `world.py`.
- Produces: `Event(zone, note)`, `AgvState(id, cell, goal_station, committed_route, route_tick)`, `WorldState(tick, clock, scenario, notes, agvs, truth)`, `Scenario(name, world, cells_per_tick, clocks, events, truth)`, `load_scenario(path) -> Scenario`, `Scenario.tick_count -> int`, `initial_agvs(world) -> dict[str, AgvState]`, `state_at(scenario, tick, agvs) -> WorldState`. `notes` maps zone id to a tuple of note strings accumulated from tick 0 through `tick` inclusive. `truth` maps every zone id to one of `clear`, `avoid`, `blocked`, defaulting to `clear`.

- [ ] **Step 1: Write the failing test**

`tests/fixtures/mini-scenario.yaml`:

```yaml
name: mini
world: mini.yaml
cells_per_tick: 2
ticks:
  - clock: "13:00"
  - clock: "13:10"
    events:
      - {zone: aisle-1, note: "13:08 Pallet wrap spill reported at the north end."}
    truth: {aisle-1: blocked}
  - clock: "13:20"
    events:
      - {zone: aisle-1, note: "13:18 Spill mopped and signed off by the shift lead."}
```

`tests/test_events.py`:

```python
from jev_costmap.events import initial_agvs, load_scenario, state_at


def scenario():
    return load_scenario("tests/fixtures/mini-scenario.yaml")


def test_scenario_loads_world_and_ticks():
    s = scenario()
    assert s.name == "mini"
    assert s.cells_per_tick == 2
    assert s.tick_count == 3
    assert s.world.width == 10


def test_notes_accumulate_across_ticks():
    s = scenario()
    agvs = initial_agvs(s.world)
    assert state_at(s, 0, agvs).notes["aisle-1"] == ()
    assert len(state_at(s, 1, agvs).notes["aisle-1"]) == 1
    notes = state_at(s, 2, agvs).notes["aisle-1"]
    assert len(notes) == 2
    assert "mopped" in notes[1]


def test_truth_defaults_to_clear_and_does_not_persist():
    s = scenario()
    agvs = initial_agvs(s.world)
    assert state_at(s, 0, agvs).truth == {"aisle-1": "clear", "aisle-2": "clear"}
    assert state_at(s, 1, agvs).truth["aisle-1"] == "blocked"
    assert state_at(s, 2, agvs).truth["aisle-1"] == "clear"


def test_clock_comes_from_the_tick():
    s = scenario()
    assert state_at(s, 1, initial_agvs(s.world)).clock == "13:10"


def test_initial_agvs_start_at_their_station_with_no_route():
    s = scenario()
    agvs = initial_agvs(s.world)
    assert agvs["agv-1"].cell == (0, 0)
    assert agvs["agv-1"].goal_station == "bay"
    assert agvs["agv-1"].committed_route == ()
    assert agvs["agv-1"].route_tick == -1


def test_states_are_independent_snapshots():
    s = scenario()
    agvs = initial_agvs(s.world)
    first = state_at(s, 1, agvs)
    state_at(s, 2, agvs)
    assert len(first.notes["aisle-1"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_events.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.events'`.

- [ ] **Step 3: Write the implementation**

`src/jev_costmap/events.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_events.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/jev_costmap/events.py tests/test_events.py tests/fixtures/mini-scenario.yaml
git commit -m "$(printf 'feat: scenario loading and per-tick world state\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 3: The state sent to Jev

**Files:**
- Create: `src/jev_costmap/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `World` from `world.py`; `WorldState` from `events.py`.
- Produces: `build_state(world: World, ws: WorldState) -> dict`. The returned dict has keys `shift`, `zones`, `agvs` and nothing else. Each zone entry has `name`, `kind`, `description`, `notes`. Each AGV entry has `position`, `committed_route`, `route_from_tick`.

The test that the truth labels never reach the state is the one protecting the whole evaluation from becoming circular. Write it first.

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:

```python
import json

from jev_costmap.events import AgvState, initial_agvs, load_scenario, state_at
from jev_costmap.state import build_state


def built(tick: int = 2):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    agvs = initial_agvs(s.world)
    agvs["agv-1"] = AgvState("agv-1", (4, 1), "bay", ("aisle-1", "aisle-2"), tick - 1)
    return build_state(s.world, state_at(s, tick, agvs))


def test_truth_labels_never_reach_the_model():
    blob = json.dumps(built())
    for label in ("blocked", "avoid", "truth"):
        assert label not in blob


def test_top_level_keys_are_exactly_these():
    assert set(built()) == {"shift", "zones", "agvs"}


def test_zone_carries_description_and_accumulated_notes():
    zone = built()["zones"]["aisle-1"]
    assert zone["name"] == "Aisle 1"
    assert zone["kind"] == "travel lane"
    assert "pallet racking" in zone["description"]
    assert len(zone["notes"]) == 2
    assert "mopped" in zone["notes"][1]


def test_agv_reports_its_zone_and_the_tick_its_route_came_from():
    agv = built(tick=2)["agvs"]["agv-1"]
    assert agv["position"] == "aisle-1"
    assert agv["committed_route"] == ["aisle-1", "aisle-2"]
    assert agv["route_from_tick"] == 1


def test_state_is_json_serialisable():
    json.dumps(built())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.state'`.

- [ ] **Step 3: Write the implementation**

`src/jev_costmap/state.py`:

```python
"""Turn a WorldState into the JSON `state` Jev receives.

The hidden ground-truth labels on a WorldState are deliberately not read here.
"""

from __future__ import annotations

from .events import WorldState
from .world import World


def build_state(world: World, ws: WorldState) -> dict:
    zones = {}
    for zone_id, zone in world.zones.items():
        zones[zone_id] = {
            "name": zone.name,
            "kind": zone.kind,
            "description": zone.description,
            "notes": list(ws.notes.get(zone_id, ())),
        }

    agvs = {}
    for agv_id, agv in ws.agvs.items():
        agvs[agv_id] = {
            "position": world.zone_at(agv.cell) or "outside any named zone",
            "committed_route": list(agv.committed_route),
            "route_from_tick": agv.route_tick,
        }

    return {
        "shift": {"name": ws.scenario, "clock": ws.clock, "tick": ws.tick},
        "zones": zones,
        "agvs": agvs,
    }
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_state.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/jev_costmap/state.py tests/test_state.py
git commit -m "$(printf 'feat: build the Jev state, with truth labels held back\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: The questions

**Files:**
- Create: `src/jev_costmap/questions.py`
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: `World` from `world.py`.
- Produces: `PEOPLE_LEVELS`, `DAMAGE_LEVELS`, `DELAY_LEVELS` (tuples of 4 strings each), `DIMENSIONS = ("people", "damage", "delay", "offlimits")`, `build_questions(world: World) -> dict[str, Score | Noul]`, `question_id(zone_id, dimension) -> str` returning `f"{zone_id}.{dimension}"`.

Questions depend only on the world's zone ids, never on the tick, so they are identical across a run and cheap to cache.

- [ ] **Step 1: Write the failing test**

`tests/test_questions.py`:

```python
from typesafe_sdk import Noul, Score

from jev_costmap.questions import (
    DAMAGE_LEVELS,
    DELAY_LEVELS,
    DIMENSIONS,
    PEOPLE_LEVELS,
    build_questions,
    question_id,
)
from jev_costmap.world import load_world_file


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_questions.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.questions'`.

- [ ] **Step 3: Write the implementation**

`src/jev_costmap/questions.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_questions.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/jev_costmap/questions.py tests/test_questions.py
git commit -m "$(printf 'feat: four Jev judgments per warehouse zone\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: Calling Jev

**Files:**
- Create: `src/jev_costmap/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime; tests use `build_questions` from `questions.py`.
- Produces: `Judgment(model, input_tokens, request_id, answers, cached)`; a real client is built with `RetryPolicy(max_retries=3)`, so transient errors retry before they ever reach the shift loop where `answers` is `dict[str, dict]` of raw JSON per question id; `MissingAnswerError`; `Judge(client=None, cache_path=None, model=None)` with `Judge.judge(state: dict, questions: Mapping[str, Score | Noul]) -> Judgment`; `cache_key(state, questions, model) -> str`.

Two traps confirmed against the SDK, both of which the implementation must handle:

1. `SystemOneResponse.request_id` **raises** `TypeSafeError` when the response carries no request id, which is the case for any response built in a test. Read it inside `try/except TypeSafeError` and fall back to `None`.
2. `NoulAnswer` has no `confidence` attribute. Never read one.

- [ ] **Step 1: Write the failing test**

`tests/test_judge.py`:

```python
import json

import pytest
from typesafe_sdk import Noul, Score, SystemOneResponse

from jev_costmap.judge import Judge, Judgment, MissingAnswerError, cache_key

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_judge.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.judge'`.

- [ ] **Step 3: Write the implementation**

`src/jev_costmap/judge.py`:

```python
"""The only module that talks to the network."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from typesafe_sdk import (
    Noul,
    RetryPolicy,
    Score,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
)

Question = Score | Noul


class SystemOneCaller(Protocol):
    def system_one(self, state: dict, questions: Mapping[str, Question], **kwargs) -> SystemOneResponse: ...


class MissingAnswerError(RuntimeError):
    """The response omitted a question we asked. That is a bug, not a condition."""


@dataclass(frozen=True)
class Judgment:
    model: str
    input_tokens: int
    request_id: str | None
    answers: dict[str, dict]
    cached: bool


def _questions_json(questions: Mapping[str, Question]) -> dict:
    return {qid: q.model_dump(mode="json") for qid, q in sorted(questions.items())}


def cache_key(state: dict, questions: Mapping[str, Question], model: str | None) -> str:
    blob = json.dumps(
        {"state": state, "questions": _questions_json(questions), "model": model},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class Judge:
    def __init__(
        self,
        client: SystemOneCaller | None = None,
        cache_path: str | Path | None = ".jev-cache.json",
        model: str | None = None,
        retry: RetryPolicy | None = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else TypeSafeClient(model=model, retry=retry or RetryPolicy(max_retries=3))
        )
        self._cache_path = Path(cache_path) if cache_path else None
        self._model = model

    def judge(self, state: dict, questions: Mapping[str, Question]) -> Judgment:
        key = cache_key(state, questions, self._model)
        cache = self._read_cache()
        if key in cache:
            stored = cache[key]
            return Judgment(
                model=stored["model"],
                input_tokens=stored["input_tokens"],
                request_id=stored["request_id"],
                answers=stored["answers"],
                cached=True,
            )

        response = self._client.system_one(state, questions)
        answers = {qid: answer.model_dump(mode="json") for qid, answer in response.answers.items()}
        missing = sorted(set(questions) - set(answers))
        if missing:
            raise MissingAnswerError(f"response omitted answers for: {', '.join(missing)}")

        try:
            request_id = response.request_id
        except TypeSafeError:
            request_id = None

        judgment = Judgment(
            model=response.model,
            input_tokens=response.usage.input_tokens or 0,
            request_id=request_id,
            answers=answers,
            cached=False,
        )
        cache[key] = {
            "model": judgment.model,
            "input_tokens": judgment.input_tokens,
            "request_id": judgment.request_id,
            "answers": judgment.answers,
        }
        self._write_cache(cache)
        return judgment

    def _read_cache(self) -> dict:
        if not self._cache_path or not self._cache_path.exists():
            return {}
        return json.loads(self._cache_path.read_text())

    def _write_cache(self, cache: dict) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_judge.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/jev_costmap/judge.py tests/test_judge.py
git commit -m "$(printf 'feat: Jev client with an on-disk judgment cache\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 6: Fusing judgments into a cost layer

**Files:**
- Create: `src/jev_costmap/costs.py`, `config/weights.yaml`
- Test: `tests/test_costs.py`

**Interfaces:**
- Consumes: `World` from `world.py`; `Judgment.answers` from `judge.py`.
- Produces: `Weights(people, damage, delay)`, `load_weights(path) -> Weights`, `DEFAULT_WEIGHTS`, `ZoneCost(zone, multiplier, blocked, detail)`, `CostLayer(grid, zones, source)`, `BLOCK_THRESHOLD = 0.5`, `BASE_COST = 1.0`, `fuse(world, answers, weights) -> dict[str, ZoneCost]`, `rasterize(world, zone_costs, source) -> CostLayer`.
- `CostLayer.grid` is `float32` shaped `(height, width)`, holding `numpy.inf` for any impassable cell. `ZoneCost.detail` maps dimension name to its normalized `0..1` value (and `offlimits` to its raw probability). `baseline.py` in Task 7 produces the same two types.

- [ ] **Step 1: Write the failing test**

`tests/test_costs.py`:

```python
import numpy as np
import pytest

from jev_costmap.costs import (
    BLOCK_THRESHOLD,
    DEFAULT_WEIGHTS,
    Weights,
    fuse,
    load_weights,
    rasterize,
)
from jev_costmap.world import load_world_file


def world():
    return load_world_file("tests/fixtures/mini.yaml")


def answers(people=0.0, damage=0.0, delay=0.0, offlimits=0.0, zones=("aisle-1", "aisle-2")):
    def score(value):
        return {
            "type": "score", "score": value, "confidence": 0.9,
            "legend": {0: "a", 1: "b", 2: "c", 3: "d"},
            "probabilities": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
        }

    out = {}
    for zone in zones:
        out[f"{zone}.people"] = score(people)
        out[f"{zone}.damage"] = score(damage)
        out[f"{zone}.delay"] = score(delay)
        out[f"{zone}.offlimits"] = {"type": "noul", "noul": offlimits}
    return out


def test_all_zero_scores_give_the_base_multiplier():
    costs = fuse(world(), answers(), DEFAULT_WEIGHTS)
    assert costs["aisle-1"].multiplier == pytest.approx(1.0)
    assert costs["aisle-1"].blocked is False


def test_maximum_scores_sum_the_weights():
    costs = fuse(world(), answers(people=3.0, damage=3.0, delay=3.0), DEFAULT_WEIGHTS)
    assert costs["aisle-1"].multiplier == pytest.approx(1.0 + 6.0 + 3.0 + 1.5)


def test_scores_are_normalised_by_the_number_of_levels():
    costs = fuse(world(), answers(people=1.5), Weights(people=2.0, damage=0.0, delay=0.0))
    assert costs["aisle-1"].detail["people"] == pytest.approx(0.5)
    assert costs["aisle-1"].multiplier == pytest.approx(2.0)


def test_the_block_threshold_is_inclusive():
    assert fuse(world(), answers(offlimits=BLOCK_THRESHOLD), DEFAULT_WEIGHTS)["aisle-1"].blocked
    assert not fuse(world(), answers(offlimits=0.49), DEFAULT_WEIGHTS)["aisle-1"].blocked


def test_detail_keeps_every_dimension():
    detail = fuse(world(), answers(people=3.0, offlimits=0.2), DEFAULT_WEIGHTS)["aisle-1"].detail
    assert detail["people"] == pytest.approx(1.0)
    assert detail["offlimits"] == pytest.approx(0.2)
    assert set(detail) == {"people", "damage", "delay", "offlimits"}


def test_a_missing_answer_is_an_error():
    partial = answers()
    del partial["aisle-2.delay"]
    with pytest.raises(KeyError):
        fuse(world(), partial, DEFAULT_WEIGHTS)


def test_rasterize_prices_cells_by_their_zone():
    costs = fuse(world(), answers(people=3.0, zones=("aisle-1",)) | answers(zones=("aisle-2",)), DEFAULT_WEIGHTS)
    layer = rasterize(world(), costs, source="jev")
    assert layer.grid[1, 3] == pytest.approx(7.0)   # aisle-1, 1 + 6*1.0
    assert layer.grid[5, 3] == pytest.approx(1.0)   # aisle-2
    assert layer.source == "jev"


def test_rasterize_marks_walls_and_blocked_zones_impassable():
    costs = fuse(world(), answers(offlimits=0.9, zones=("aisle-1",)) | answers(zones=("aisle-2",)), DEFAULT_WEIGHTS)
    layer = rasterize(world(), costs, source="jev")
    assert np.isinf(layer.grid[1, 3])   # blocked zone
    assert np.isinf(layer.grid[3, 3])   # wall
    assert np.isfinite(layer.grid[3, 4])  # the doorway punched through the wall
    assert layer.grid[3, 4] == pytest.approx(1.0)  # no zone: base cost


def test_weights_load_from_yaml(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text("people: 5.0\ndamage: 2.0\ndelay: 1.0\n")
    assert load_weights(path) == Weights(people=5.0, damage=2.0, delay=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_costs.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.costs'`.

- [ ] **Step 3: Write the config**

`config/weights.yaml`:

```yaml
# Starting values. Tune against the scenarios; the demo does not depend on
# these particular numbers.
people: 6.0
damage: 3.0
delay: 1.5
```

- [ ] **Step 4: Write the implementation**

`src/jev_costmap/costs.py`:

```python
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
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_costs.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/jev_costmap/costs.py config/weights.yaml tests/test_costs.py
git commit -m "$(printf 'feat: fuse judgments into a rasterized cost layer\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 7: The keyword rule baseline

**Files:**
- Create: `src/jev_costmap/baseline.py`, `config/rules.yaml`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `World`; `WorldState` from `events.py`; `ZoneCost`, `CostLayer`, `rasterize` from `costs.py`.
- Produces: `Rule(pattern, multiplier, blocked)`, `load_rules(path) -> tuple[Rule, ...]`, `DEFAULT_RULES`, `baseline_costs(world, ws, rules) -> dict[str, ZoneCost]`, `baseline_layer(world, ws, rules) -> CostLayer` with `source="baseline"`.
- Matching text for a zone is its description plus every accumulated note, joined by newlines and lowercased. A block from any matching rule wins outright; otherwise matching multipliers are multiplied together. `ZoneCost.detail` holds `{pattern: multiplier}` for each matching rule, so `disagree` can name what fired.

- [ ] **Step 1: Write the failing test**

`tests/test_baseline.py`:

```python
import numpy as np
import pytest

from jev_costmap.baseline import DEFAULT_RULES, Rule, baseline_costs, baseline_layer, load_rules
from jev_costmap.events import WorldState
from jev_costmap.world import load_world_file


def world():
    return load_world_file("tests/fixtures/mini.yaml")


def state(notes: dict[str, tuple[str, ...]]) -> WorldState:
    return WorldState(
        tick=0, clock="13:00", scenario="t",
        notes={"aisle-1": (), "aisle-2": (), **notes},
        agvs={}, truth={},
    )


def test_a_zone_with_no_matching_note_costs_nothing_extra():
    costs = baseline_costs(world(), state({}), DEFAULT_RULES)
    assert costs["aisle-1"].multiplier == pytest.approx(1.0)
    assert costs["aisle-1"].blocked is False


def test_a_blocking_keyword_blocks():
    costs = baseline_costs(world(), state({"aisle-1": ("13:31 Pallet wrap spill reported.",)}), DEFAULT_RULES)
    assert costs["aisle-1"].blocked is True


def test_a_correction_cannot_unblock_a_keyword_match():
    """The baseline's defining weakness, pinned as a test."""
    notes = ("13:31 Pallet wrap spill reported.", "13:40 Spill mopped and signed off.")
    assert baseline_costs(world(), state({"aisle-1": notes}), DEFAULT_RULES)["aisle-1"].blocked is True


def test_wording_with_no_keyword_is_invisible_to_the_baseline():
    notes = ("13:38 Stock count under way; the counter steps in and out of the lane.",)
    costs = baseline_costs(world(), state({"aisle-1": notes}), DEFAULT_RULES)
    assert costs["aisle-1"].multiplier == pytest.approx(1.0)


def test_multipliers_from_several_rules_compound():
    rules = (Rule("picker", 3.0, False), Rule("fragile", 2.0, False))
    costs = baseline_costs(world(), state({"aisle-1": ("a picker near fragile glass",)}), rules)
    assert costs["aisle-1"].multiplier == pytest.approx(6.0)
    assert set(costs["aisle-1"].detail) == {"picker", "fragile"}


def test_a_block_wins_over_multipliers_regardless_of_order():
    rules = (Rule("picker", 3.0, False), Rule("spill", None, True))
    costs = baseline_costs(world(), state({"aisle-1": ("a picker slipped on a spill",)}), rules)
    assert costs["aisle-1"].blocked is True


def test_matching_is_case_insensitive_and_reads_the_description_too():
    costs = baseline_costs(world(), state({}), (Rule("pallet racking", 4.0, False),))
    assert costs["aisle-1"].multiplier == pytest.approx(4.0)


def test_baseline_layer_rasterizes_like_the_jev_layer():
    layer = baseline_layer(world(), state({"aisle-1": ("a spill",)}), DEFAULT_RULES)
    assert layer.source == "baseline"
    assert np.isinf(layer.grid[1, 3])
    assert layer.grid[5, 3] == pytest.approx(1.0)


def test_rules_load_from_yaml(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text("- {pattern: 'spill|leak', blocked: true}\n- {pattern: picker, multiplier: 3.0}\n")
    rules = load_rules(path)
    assert rules[0] == Rule("spill|leak", None, True)
    assert rules[1] == Rule("picker", 3.0, False)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_baseline.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.baseline'`.

- [ ] **Step 3: Write the config**

`config/rules.yaml`:

```yaml
# A keyword cost model of the kind that ships in real dispatch systems. It is
# meant to be a fair opponent: it gets the obvious cases right.
- {pattern: 'spill|wet floor|leak|leaking', blocked: true}
- {pattern: 'closed|do not enter|no entry|out of service', blocked: true}
- {pattern: 'picker|pickers|staff|operator|person|people|engineer', multiplier: 3.0}
- {pattern: 'fragile|glass|glassware|breakable', multiplier: 2.0}
- {pattern: 'obstruction|blocked aisle|pallet left', multiplier: 2.0}
```

- [ ] **Step 4: Write the implementation**

`src/jev_costmap/baseline.py`:

```python
"""A keyword cost model, planned alongside Jev so the demo can be checked.

It is not a strawman: it catches every case someone thought to add a word for.
What it cannot do is read a correction, a time, or a spatial qualifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_baseline.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add src/jev_costmap/baseline.py config/rules.yaml tests/test_baseline.py
git commit -m "$(printf 'feat: keyword rule baseline over the same world states\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 8: Space-time A\* for one AGV

**Files:**
- Create: `src/jev_costmap/planner.py`, `tests/helpers.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `World`, `Cell` from `world.py`; `CostLayer` from `costs.py`.
- Produces: `WAIT_COST = 0.5`, `HORIZON = 400`, `Plan(agv, path, cost, deferred)` where `path` is a tuple of one cell per timestep starting at the AGV's current cell; `Reservations(horizon=HORIZON)` with `.add(path)`, `.vertex(cell, t) -> bool`, `.edge(frm, to, t) -> bool`; `plan_single(world, layer, agv, start, goal, reservations=None, horizon=HORIZON) -> Plan | None`.
- Also creates `tests/helpers.py` with `grid_world(width, height, lanes)` and `uniform_layer(world, multipliers=None, blocked=())`, used by this task and Task 9.

Two rules the implementation must get right, because both look fine on a rendered frame when wrong:

- **Edge conflicts.** Two AGVs swapping cells between consecutive timesteps never collide in vertex terms but pass through each other. `Reservations.edge` exists for this.
- **Parked AGVs.** An AGV that reaches its goal stays there. Its goal cell must be reserved for every timestep from arrival to the horizon, not just for the length of its path.

- [ ] **Step 1: Write the test helpers**

`tests/helpers.py`:

```python
"""Small synthetic worlds and cost layers for planner tests."""

from jev_costmap.costs import CostLayer, ZoneCost, rasterize
from jev_costmap.world import World, load_world


def grid_world(width: int, height: int) -> World:
    """One horizontal lane zone per row, named row-0, row-1, ..."""
    return load_world(
        {
            "width": width,
            "height": height,
            "cell_size_m": 0.5,
            "zones": [
                {
                    "id": f"row-{y}",
                    "name": f"Row {y}",
                    "kind": "travel lane",
                    "description": "Travel lane between pallet racking.",
                    "x": 0,
                    "y": y,
                    "w": width,
                    "h": 1,
                }
                for y in range(height)
            ],
            "stations": [
                {"id": "west", "cell": [0, 0]},
                {"id": "east", "cell": [width - 1, 0]},
            ],
            "agvs": [
                {"id": "agv-1", "priority": 1, "start": "west", "tasks": ["east"]},
                {"id": "agv-2", "priority": 2, "start": "east", "tasks": ["west"]},
            ],
        }
    )


def uniform_layer(world: World, multipliers=None, blocked=()) -> CostLayer:
    multipliers = multipliers or {}
    costs = {
        zone_id: ZoneCost(
            zone=zone_id,
            multiplier=float(multipliers.get(zone_id, 1.0)),
            blocked=zone_id in blocked,
            detail={},
        )
        for zone_id in world.zones
    }
    return rasterize(world, costs, source="test")
```

- [ ] **Step 2: Write the failing test**

`tests/test_planner.py`:

```python
import pytest

from jev_costmap.planner import HORIZON, Plan, Reservations, plan_single
from tests.helpers import grid_world, uniform_layer


def test_reservations_record_every_vertex():
    r = Reservations()
    r.add(((0, 0), (1, 0), (2, 0)))
    assert r.vertex((1, 0), 1)
    assert not r.vertex((1, 0), 0)


def test_a_parked_agv_holds_its_goal_to_the_horizon():
    r = Reservations()
    r.add(((0, 0), (1, 0)))
    assert r.vertex((1, 0), 1)
    assert r.vertex((1, 0), 50)
    assert r.vertex((1, 0), HORIZON)


def test_reservations_detect_a_swap():
    r = Reservations()
    r.add(((4, 0), (3, 0), (2, 0)))
    assert r.edge((3, 0), (4, 0), 1)      # we would move east into their westward move
    assert not r.edge((3, 0), (4, 0), 2)


def test_a_clear_lane_gives_the_direct_path():
    w = grid_world(5, 1)
    plan = plan_single(w, uniform_layer(w), "agv-1", (0, 0), (4, 0))
    assert plan.path == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    assert plan.cost == pytest.approx(4.0)
    assert plan.deferred is False


def test_an_expensive_lane_is_detoured_around():
    w = grid_world(5, 2)
    layer = uniform_layer(w, multipliers={"row-1": 10.0})
    plan = plan_single(w, layer, "agv-1", (0, 1), (4, 1))
    assert (2, 0) in plan.path
    assert plan.cost < 4 * 10.0


def test_a_blocked_zone_is_never_entered():
    w = grid_world(5, 2)
    layer = uniform_layer(w, blocked=("row-0",))
    plan = plan_single(w, layer, "agv-1", (0, 1), (4, 1))
    assert all(cell[1] == 1 for cell in plan.path)


def test_a_blocked_goal_has_no_plan():
    w = grid_world(5, 2)
    layer = uniform_layer(w, blocked=("row-0",))
    assert plan_single(w, layer, "agv-1", (0, 1), (4, 0)) is None


def test_an_agv_starting_in_a_newly_blocked_zone_can_still_leave():
    w = grid_world(5, 2)
    layer = uniform_layer(w, blocked=("row-0",))
    plan = plan_single(w, layer, "agv-1", (0, 0), (4, 1))
    assert plan is not None
    assert plan.path[0] == (0, 0)
    assert all(cell[1] == 1 for cell in plan.path[1:])


def test_a_reserved_cell_makes_the_agv_wait():
    w = grid_world(5, 1)
    r = Reservations()
    r._vertex.add(((1, 0), 1))       # a bare vertex hold, no parked path
    plan = plan_single(w, uniform_layer(w), "agv-1", (0, 0), (4, 0), reservations=r)
    assert plan.path[1] == (0, 0)
    assert len(plan.path) == 6
    assert plan.cost == pytest.approx(4.0 + 0.5)


def test_an_agv_parked_across_a_one_wide_lane_blocks_it_for_good():
    """The other AGV stops in the middle and stays there, so there is no route
    and no amount of waiting produces one. Reserving a parked goal only to the
    end of its path instead of to the horizon would wrongly find a plan here."""
    w = grid_world(5, 1)
    r = Reservations()
    r.add(((0, 0), (1, 0), (2, 0)))
    assert plan_single(w, uniform_layer(w), "agv-2", (4, 0), (0, 0), reservations=r) is None


def test_two_agvs_can_pass_in_a_two_wide_lane():
    w = grid_world(5, 2)
    r = Reservations()
    r.add(((4, 0), (3, 0), (2, 0), (1, 0), (0, 0)))
    plan = plan_single(w, uniform_layer(w), "agv-2", (0, 1), (4, 1), reservations=r)
    assert plan is not None


def test_the_horizon_is_respected():
    w = grid_world(5, 1)
    assert plan_single(w, uniform_layer(w), "agv-1", (0, 0), (4, 0), horizon=2) is None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_planner.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.planner'`.

- [ ] **Step 4: Write the implementation**

`src/jev_costmap/planner.py`:

```python
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
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_planner.py -v
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/jev_costmap/planner.py tests/test_planner.py tests/helpers.py
git commit -m "$(printf 'feat: space-time A* with vertex and edge reservations\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 9: Prioritized planning for all AGVs

**Files:**
- Modify: `src/jev_costmap/planner.py` (append; do not change `plan_single`)
- Test: `tests/test_plan_all.py`

**Interfaces:**
- Consumes: `Plan`, `Reservations`, `plan_single` from Task 8.
- Produces: `PlanRequest(agv, priority, start, goal)`; `plan_all(world, layer, requests, horizon=HORIZON) -> dict[str, Plan]`. Requests are planned in ascending `priority` (1 is highest). An AGV with no plan gets `Plan(agv, path=(start,), cost=0.0, deferred=True)` and leaves no reservation, so it holds position and retries next tick.

- [ ] **Step 1: Write the failing test**

`tests/test_plan_all.py`:

```python
import pytest

from jev_costmap.planner import PlanRequest, plan_all
from tests.helpers import grid_world, uniform_layer


def assert_no_conflicts(plans):
    """No two AGVs share a cell at a timestep, and none swap through each other."""
    paths = {p.agv: p.path for p in plans.values()}
    span = max(len(p) for p in paths.values())
    for t in range(span):
        at = {}
        for agv, path in paths.items():
            cell = path[min(t, len(path) - 1)]
            assert cell not in at, f"{agv} and {at[cell]} share {cell} at t={t}"
            at[cell] = agv
        if t == 0:
            continue
        for a, pa in paths.items():
            for b, pb in paths.items():
                if a >= b:
                    continue
                a_prev, a_now = pa[min(t - 1, len(pa) - 1)], pa[min(t, len(pa) - 1)]
                b_prev, b_now = pb[min(t - 1, len(pb) - 1)], pb[min(t, len(pb) - 1)]
                assert not (a_prev == b_now and b_prev == a_now), f"{a} and {b} swapped at t={t}"


def test_the_edge_rule_holds_end_to_end():
    """`Reservations.edge` is unit-tested above; this is the behavioural check
    that plan_all never produces a swap, which no vertex check would catch."""
    w = grid_world(5, 2)
    plans = plan_all(
        w,
        uniform_layer(w),
        [PlanRequest("agv-1", 1, (0, 0), (4, 0)), PlanRequest("agv-2", 2, (4, 0), (0, 1))],
    )
    assert_no_conflicts(plans)


def test_two_agvs_pass_each_other_in_a_two_wide_lane():
    w = grid_world(5, 2)
    plans = plan_all(
        w,
        uniform_layer(w),
        [
            PlanRequest("agv-1", 1, (0, 0), (4, 0)),
            PlanRequest("agv-2", 2, (4, 0), (0, 0)),
        ],
    )
    assert not plans["agv-1"].deferred
    assert not plans["agv-2"].deferred
    assert_no_conflicts(plans)


def test_the_highest_priority_agv_gets_the_direct_path():
    w = grid_world(5, 2)
    plans = plan_all(
        w,
        uniform_layer(w),
        [
            PlanRequest("agv-2", 2, (4, 0), (0, 0)),
            PlanRequest("agv-1", 1, (0, 0), (4, 0)),
        ],
    )
    assert plans["agv-1"].path == ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))


def test_a_starved_agv_is_deferred_and_holds_position():
    # agv-1 parks in the middle of a one-wide lane, so agv-2 has no route at all.
    w = grid_world(5, 1)
    plans = plan_all(
        w,
        uniform_layer(w),
        [
            PlanRequest("agv-1", 1, (0, 0), (2, 0)),
            PlanRequest("agv-2", 2, (4, 0), (0, 0)),
        ],
    )
    assert plans["agv-1"].deferred is False
    assert plans["agv-2"].deferred is True
    assert plans["agv-2"].path == ((4, 0),)
    assert plans["agv-2"].cost == pytest.approx(0.0)


def test_every_requested_agv_appears_in_the_result():
    w = grid_world(5, 1)
    requests = [PlanRequest("agv-1", 1, (0, 0), (4, 0)), PlanRequest("agv-2", 2, (4, 0), (0, 0))]
    assert set(plan_all(w, uniform_layer(w), requests)) == {"agv-1", "agv-2"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_plan_all.py -v
```

Expected: `ImportError: cannot import name 'PlanRequest'`.

- [ ] **Step 3: Append the implementation to `planner.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_planner.py tests/test_plan_all.py -v
```

Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/jev_costmap/planner.py tests/test_plan_all.py
git commit -m "$(printf 'feat: prioritized multi-AGV planning with deferrals\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 10: The run directory

**Files:**
- Create: `src/jev_costmap/runs.py`
- Test: `tests/test_runs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunDir(root)` with `RunDir.create(base, scenario_name, when=None)`, `RunDir.open(path)`, `.write_tick(kind, tick, data)`, `.read_tick(kind, tick)`, `.write_manifest(data)`, `.read_manifest()`, `.ticks -> list[int]`, `.frames_dir -> Path`, `.gif_path -> Path`, `.report_path -> Path`. `kind` is one of `states`, `questions`, `answers`, `costs`, `plans`, `truth`; files are `<kind>/tNN.json`.

**Deviation from the spec, deliberate:** the spec lists `costs/t00.npz`. This plan stores `costs/tNN.json` holding the per-zone multipliers, blocked flags and detail for both sources instead, and re-rasterizes the grid on replay via `costs.rasterize`, which is deterministic. The artifacts stay human-readable and diffable, and nothing downstream needs the dense array on disk.

- [ ] **Step 1: Write the failing test**

`tests/test_runs.py`:

```python
import datetime as dt

import pytest

from jev_costmap.runs import RunDir


def test_create_names_the_directory_by_time_and_scenario(tmp_path):
    run = RunDir.create(tmp_path, "night-shift", when=dt.datetime(2026, 9, 21, 14, 2))
    assert run.root.name == "2026-09-21T14-02-night-shift"
    assert run.root.is_dir()


def test_tick_files_round_trip(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    run.write_tick("answers", 7, {"aisle-1.people": {"score": 2.0}})
    assert (run.root / "answers" / "t07.json").exists()
    assert run.read_tick("answers", 7)["aisle-1.people"]["score"] == 2.0


def test_ticks_are_discovered_in_order(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    for tick in (2, 0, 11, 1):
        run.write_tick("states", tick, {})
    assert run.ticks == [0, 1, 2, 11]


def test_manifest_round_trips(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    run.write_manifest({"model": "jev-1.13.0", "input_tokens": 1821})
    assert RunDir.open(run.root).read_manifest()["model"] == "jev-1.13.0"


def test_an_unknown_kind_is_rejected(tmp_path):
    run = RunDir.create(tmp_path, "s", when=dt.datetime(2026, 9, 21, 14, 2))
    with pytest.raises(ValueError, match="unknown kind"):
        run.write_tick("nonsense", 0, {})


def test_open_rejects_a_directory_with_no_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunDir.open(tmp_path / "nope").read_manifest()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_runs.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.runs'`.

- [ ] **Step 3: Write the implementation**

`src/jev_costmap/runs.py`:

```python
"""On-disk layout of a run. Everything a replay needs lives here."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

KINDS = ("states", "questions", "answers", "costs", "plans", "truth")


@dataclass(frozen=True)
class RunDir:
    root: Path

    @classmethod
    def create(cls, base: str | Path, scenario_name: str, when: dt.datetime | None = None) -> RunDir:
        when = when or dt.datetime.now()
        root = Path(base) / f"{when:%Y-%m-%dT%H-%M}-{scenario_name}"
        root.mkdir(parents=True, exist_ok=True)
        for kind in KINDS:
            (root / kind).mkdir(exist_ok=True)
        (root / "frames").mkdir(exist_ok=True)
        return cls(root)

    @classmethod
    def open(cls, path: str | Path) -> RunDir:
        return cls(Path(path))

    def _path(self, kind: str, tick: int) -> Path:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
        return self.root / kind / f"t{tick:02d}.json"

    def write_tick(self, kind: str, tick: int, data) -> None:
        path = self._path(kind, tick)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, default=str))

    def read_tick(self, kind: str, tick: int):
        return json.loads(self._path(kind, tick).read_text())

    def write_manifest(self, data: dict) -> None:
        (self.root / "manifest.json").write_text(json.dumps(data, indent=1, default=str))

    def read_manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    @property
    def ticks(self) -> list[int]:
        return sorted(int(p.stem[1:]) for p in (self.root / "states").glob("t*.json"))

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def gif_path(self) -> Path:
        return self.root / "shift.gif"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_runs.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/jev_costmap/runs.py tests/test_runs.py
git commit -m "$(printf 'feat: run directory and manifest\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 11: The shift loop and `run`

**Files:**
- Create: `src/jev_costmap/shift.py`, `src/jev_costmap/cli.py`
- Test: `tests/test_shift.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1-10.
- Produces: `TickResult(tick, state, judgment, jev_costs, baseline_costs, jev_layer, baseline_layer, jev_plans, baseline_plans)`; `advance(scenario, agvs, plans, tick) -> dict[str, AgvState]`; `run_shift(scenario, judge, run, weights, rules) -> dict` returning the manifest it wrote; `ShiftAborted`; `cli.main(argv=None) -> int`.

Rules the implementation must follow:

- Both planners plan from **identical** AGV positions each tick. Only the Jev plans drive the simulation forward; the baseline plans exist to be scored. Anything else makes the comparison meaningless.
- `advance` moves each AGV `min(cells_per_tick, len(path) - 1)` steps along its Jev path, sets `committed_route` to the zone ids the *remaining* path passes through in order (deduplicated, consecutive repeats collapsed), and sets `route_tick` to the tick just planned. That route reaches Jev at the *next* tick, which is the whole point.
- An AGV that reaches its goal station takes the next task in its list, wrapping around.
- A deferred AGV does not move.
- **A final API failure aborts the run and is recorded.** The SDK has already
  retried by the time an error reaches `run_shift`. Catch `TypeSafeAPIError`,
  append `{"tick": n, "status": "failed", "error": ...}` to the manifest, write
  the manifest so every artifact produced so far stays usable, then raise
  `ShiftAborted`. Never fall back to the baseline.

- [ ] **Step 1: Write the failing test**

`tests/test_shift.py`:

```python
import datetime as dt

import pytest
from typesafe_sdk import SystemOneResponse, TypeSafeAPIError

from jev_costmap.baseline import DEFAULT_RULES
from jev_costmap.costs import DEFAULT_WEIGHTS
from jev_costmap.events import initial_agvs, load_scenario
from jev_costmap.judge import Judge
from jev_costmap.planner import Plan
from jev_costmap.runs import RunDir
from jev_costmap.shift import ShiftAborted, advance, run_shift


class CalmClient:
    """Answers every question with 'nothing going on here'."""

    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kwargs):
        self.calls += 1
        answers = {}
        for qid in questions:
            if qid.endswith(".offlimits"):
                answers[qid] = {"type": "noul", "noul": 0.02}
            else:
                answers[qid] = {
                    "type": "score", "score": 0.0, "confidence": 0.95,
                    "legend": {0: "a", 1: "b", 2: "c", 3: "d"},
                    "probabilities": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0},
                }
        return SystemOneResponse.model_validate(
            {"model": "jev-1.13.0", "usage": {"input_tokens": 500}, "answers": answers}
        )


def scenario():
    return load_scenario("tests/fixtures/mini-scenario.yaml")


def test_advance_moves_the_agv_and_records_the_route_tick():
    s = scenario()
    agvs = initial_agvs(s.world)
    plans = {"agv-1": Plan("agv-1", ((0, 0), (1, 0), (2, 0), (3, 0)), 3.0, False)}
    moved = advance(s, agvs, plans, tick=0)
    assert moved["agv-1"].cell == (2, 0)          # cells_per_tick is 2
    assert moved["agv-1"].route_tick == 0
    assert moved["agv-1"].committed_route == ("aisle-1",)


def test_a_deferred_agv_does_not_move():
    s = scenario()
    agvs = initial_agvs(s.world)
    plans = {"agv-1": Plan("agv-1", ((0, 0),), 0.0, True)}
    assert advance(s, agvs, plans, tick=0)["agv-1"].cell == (0, 0)


def test_run_shift_writes_one_file_per_kind_per_tick(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    judge = Judge(client=CalmClient(), cache_path=tmp_path / "c.json")
    run_shift(s, judge, run, DEFAULT_WEIGHTS, DEFAULT_RULES)

    assert run.ticks == [0, 1, 2]
    for tick in run.ticks:
        for kind in ("states", "questions", "answers", "costs", "plans", "truth"):
            run.read_tick(kind, tick)


def test_the_manifest_records_the_model_tokens_and_weights(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    m = run.read_manifest()
    assert m["model"] == "jev-1.13.0"
    assert m["input_tokens"] == 1500
    assert m["weights"] == {"people": 6.0, "damage": 3.0, "delay": 1.5}
    assert all(t["status"] == "ok" for t in m["ticks"])


def test_stored_plans_carry_both_sources(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    plans = run.read_tick("plans", 1)
    assert set(plans) == {"jev", "baseline"}
    assert "agv-1" in plans["jev"]


def test_the_second_tick_tells_jev_the_route_from_the_first(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert run.read_tick("states", 1)["agvs"]["agv-1"]["route_from_tick"] == 0


class FailingClient(CalmClient):
    """Answers tick 0, then fails the way the SDK does once retries are spent."""

    def system_one(self, state, questions, **kwargs):
        if self.calls >= 1:
            raise TypeSafeAPIError("503 upstream unavailable")
        return super().system_one(state, questions, **kwargs)


def test_a_failed_tick_aborts_the_run_and_keeps_what_was_written(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    judge = Judge(client=FailingClient(), cache_path=tmp_path / "c.json")
    with pytest.raises(ShiftAborted):
        run_shift(s, judge, run, DEFAULT_WEIGHTS, DEFAULT_RULES)

    assert run.ticks == [0]                       # tick 0's artifacts survived
    entries = run.read_manifest()["ticks"]
    assert entries[0]["status"] == "ok"
    assert entries[1]["status"] == "failed"
    assert "503" in entries[1]["error"]


def test_a_failure_never_falls_back_to_the_baseline(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    with pytest.raises(ShiftAborted):
        run_shift(s, Judge(client=FailingClient(), cache_path=tmp_path / "c.json"),
                  run, DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert 1 not in run.ticks


def test_truth_is_stored_for_scoring_but_not_in_the_state(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert run.read_tick("truth", 1)["aisle-1"] == "blocked"
    assert "blocked" not in str(run.read_tick("states", 1))
```

`tests/test_cli.py`:

```python
import json

from jev_costmap.cli import main


def test_dry_run_prints_the_state_and_questions_and_makes_no_call(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["run", "--scenario", "tests/fixtures/mini-scenario.yaml", "--dry-run"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"state", "questions"}
    assert printed["state"]["shift"]["tick"] == 0
    assert len(printed["questions"]) == 8


def test_run_without_an_api_key_fails_with_a_useful_message(capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code = main(["run", "--scenario", "tests/fixtures/mini-scenario.yaml", "--out", str(tmp_path)])
    assert code == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_unknown_command_is_an_error():
    assert main(["nonsense"]) != 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_shift.py tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.shift'`.

- [ ] **Step 3: Write `shift.py`**

```python
"""One tick: judge the scene, price the map, plan every AGV, advance."""

from __future__ import annotations

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


class ShiftAborted(RuntimeError):
    """A tick failed for good. Artifacts already written stay on disk."""


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

    def finish(failure: dict | None = None) -> dict:
        if failure:
            tick_entries.append(failure)
        manifest = {
            "scenario": scenario.name,
            "model": model,
            "input_tokens": total_tokens,
            "estimated_usd": round(total_tokens * 0.042 / 1_000_000, 6),
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
```

- [ ] **Step 4: Write `cli.py` with the `run` command only**

Later tasks add `replay`, `explain` and `disagree` subparsers to `_parser()`.

```python
"""The jev-costmap command line."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .baseline import load_rules
from .costs import load_weights
from .events import initial_agvs, load_scenario, state_at
from .judge import Judge
from .questions import build_questions
from .runs import RunDir
from .shift import ShiftAborted, run_shift
from .state import build_state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-costmap")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan a whole shift")
    run.add_argument("--scenario", required=True)
    run.add_argument("--out", default="runs")
    run.add_argument("--weights", default="config/weights.yaml")
    run.add_argument("--rules", default="config/rules.yaml")
    run.add_argument("--model", default=None)
    run.add_argument("--dry-run", action="store_true",
                     help="print tick 0's state and questions; make no API call")
    return parser


def _cmd_run(args) -> int:
    scenario = load_scenario(args.scenario)
    if args.dry_run:
        ws = state_at(scenario, 0, initial_agvs(scenario.world))
        print(json.dumps({
            "state": build_state(scenario.world, ws),
            "questions": {
                qid: q.model_dump(mode="json")
                for qid, q in build_questions(scenario.world).items()
            },
        }, indent=1))
        return 0

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set. Create a key at https://console.typesafe.ai/ "
              "and export it, or use --dry-run.", file=sys.stderr)
        return 2

    run = RunDir.create(args.out, scenario.name)
    try:
        manifest = run_shift(
            scenario,
            Judge(model=args.model),
            run,
            load_weights(args.weights),
            load_rules(args.rules),
        )
    except ShiftAborted as aborted:
        print(f"{aborted}. Artifacts written so far are in {run.root}.", file=sys.stderr)
        return 1
    print(f"{run.root}  {manifest['input_tokens']} tokens  ~${manifest['estimated_usd']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return {"run": _cmd_run}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_shift.py tests/test_cli.py -v
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/jev_costmap/shift.py src/jev_costmap/cli.py tests/test_shift.py tests/test_cli.py
git commit -m "$(printf 'feat: shift loop and the run command\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 12: Frames, GIF and `replay`

**Files:**
- Create: `src/jev_costmap/render.py`
- Modify: `src/jev_costmap/events.py` (add `path` to `Scenario`), `src/jev_costmap/shift.py` (record `scenario_path` in the manifest), `src/jev_costmap/cli.py` (add the `replay` subcommand), `tests/test_events.py` (one new assertion), `tests/test_shift.py` (one new assertion)
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `World`, `CostLayer`, `rasterize`, `ZoneCost`, `RunDir`, `Scenario`.
- Produces: `layers_from_run(world, run, tick) -> tuple[CostLayer, CostLayer]`, `paths_from_run(run, tick) -> dict[str, dict[str, list[Cell]]]`, `render_tick(scenario, run, tick) -> Path`, `render_run(scenario, run) -> Path` (returns the GIF path).

Each frame is two panels side by side — Jev's cost layer with Jev's paths on the left, the baseline's with the baseline's paths on the right — so the comparison is visible in every single frame rather than only in the report.

- [ ] **Step 1: Make the scenario path recoverable**

In `events.py`, add `path: Path` as the last field of `Scenario` and set it in `load_scenario`:

```python
@dataclass(frozen=True)
class Scenario:
    name: str
    world: World
    cells_per_tick: int
    clocks: tuple[str, ...]
    events: tuple[tuple[Event, ...], ...]
    truth: tuple[Mapping[str, str], ...]
    path: Path
```

and in the `return Scenario(...)` call add `path=path`.

Add to `tests/test_events.py`:

```python
def test_scenario_remembers_where_it_was_loaded_from():
    assert scenario().path.name == "mini-scenario.yaml"
```

In `shift.py`, add one line to the manifest dict inside `finish()`:

```python
        "scenario_path": str(scenario.path),
```

Add to `tests/test_shift.py`:

```python
def test_the_manifest_records_the_scenario_path(tmp_path):
    s = scenario()
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    assert run.read_manifest()["scenario_path"].endswith("mini-scenario.yaml")
```

- [ ] **Step 2: Write the failing test**

`tests/test_render.py`:

```python
import datetime as dt

from jev_costmap.baseline import DEFAULT_RULES
from jev_costmap.costs import DEFAULT_WEIGHTS
from jev_costmap.events import load_scenario
from jev_costmap.judge import Judge
from jev_costmap.render import layers_from_run, render_run, render_tick
from jev_costmap.runs import RunDir
from jev_costmap.shift import run_shift
from tests.test_shift import CalmClient


def recorded(tmp_path):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    return s, run


def test_layers_are_rebuilt_from_stored_costs(tmp_path):
    s, run = recorded(tmp_path)
    jev, baseline = layers_from_run(s.world, run, 1)
    assert jev.source == "jev"
    assert baseline.source == "baseline"
    assert jev.grid.shape == (s.world.height, s.world.width)


def test_a_frame_is_written_per_tick(tmp_path):
    s, run = recorded(tmp_path)
    frame = render_tick(s, run, 1)
    assert frame.exists()
    assert frame.stat().st_size > 5000


def test_render_run_produces_a_gif_and_every_frame(tmp_path):
    s, run = recorded(tmp_path)
    gif = render_run(s, run)
    assert gif.exists()
    assert gif.stat().st_size > 5000
    assert len(list(run.frames_dir.glob("t*.png"))) == 3


def test_replay_needs_no_api_key(tmp_path, monkeypatch, capsys):
    from jev_costmap.cli import main

    s, run = recorded(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["replay", str(run.root)]) == 0
    assert run.gif_path.exists()
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_render.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.render'`.

- [ ] **Step 4: Write the implementation**

`src/jev_costmap/render.py`:

```python
"""Frames and a GIF, rebuilt from stored artifacts. No network, no API key."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from PIL import Image  # noqa: E402

from .costs import CostLayer, ZoneCost, rasterize  # noqa: E402
from .events import Scenario  # noqa: E402
from .runs import RunDir  # noqa: E402
from .world import World  # noqa: E402

AGV_COLOURS = ("#e8543f", "#2f7fd0", "#f0a202", "#3f9d59")


def layers_from_run(world: World, run: RunDir, tick: int) -> tuple[CostLayer, CostLayer]:
    stored = run.read_tick("costs", tick)
    layers = []
    for source in ("jev", "baseline"):
        zone_costs = {
            zone: ZoneCost(
                zone=zone,
                multiplier=float(entry["multiplier"]),
                blocked=bool(entry["blocked"]),
                detail=entry["detail"],
            )
            for zone, entry in stored[source].items()
        }
        layers.append(rasterize(world, zone_costs, source=source))
    return layers[0], layers[1]


def paths_from_run(run: RunDir, tick: int) -> dict[str, dict[str, list[tuple[int, int]]]]:
    stored = run.read_tick("plans", tick)
    return {
        source: {agv: [tuple(c) for c in plan["path"]] for agv, plan in agvs.items()}
        for source, agvs in stored.items()
    }


def _panel(ax, world: World, layer: CostLayer, paths, title: str) -> None:
    display = np.array(layer.grid, dtype=float)
    display[world.obstacles] = np.nan
    blocked = np.isinf(display)
    display[blocked] = np.nan

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d9d9d9")
    ax.imshow(display, cmap=cmap, norm=LogNorm(vmin=1.0, vmax=12.0), origin="upper")

    overlay = np.zeros((*display.shape, 4))
    overlay[blocked] = (0.35, 0.0, 0.05, 0.85)
    overlay[world.obstacles] = (0.15, 0.15, 0.15, 1.0)
    ax.imshow(overlay, origin="upper")

    for index, (agv, path) in enumerate(sorted(paths.items())):
        xs = [c[0] for c in path]
        ys = [c[1] for c in path]
        colour = AGV_COLOURS[index % len(AGV_COLOURS)]
        ax.plot(xs, ys, color=colour, linewidth=1.8, label=agv)
        ax.plot(xs[0], ys[0], "o", color=colour, markersize=5)

    for zone in world.zones.values():
        ax.text(zone.x + zone.w / 2, zone.y + zone.h / 2, zone.id,
                ha="center", va="center", fontsize=6, color="#222222")

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def render_tick(scenario: Scenario, run: RunDir, tick: int) -> Path:
    world = scenario.world
    jev, baseline = layers_from_run(world, run, tick)
    paths = paths_from_run(run, tick)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _panel(axes[0], world, jev, paths["jev"], "Jev semantic cost layer")
    _panel(axes[1], world, baseline, paths["baseline"], "Keyword rule baseline")

    notes = " | ".join(event.note for event in scenario.events[tick]) or "no new events"
    fig.suptitle(f"{scenario.name}  t={tick}  {scenario.clocks[tick]}", fontsize=12)
    fig.text(0.5, 0.02, notes, ha="center", fontsize=9, wrap=True)
    axes[0].legend(loc="upper right", fontsize=7)

    run.frames_dir.mkdir(parents=True, exist_ok=True)
    out = run.frames_dir / f"t{tick:02d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def render_run(scenario: Scenario, run: RunDir) -> Path:
    for tick in run.ticks:
        render_tick(scenario, run, tick)
    frames = [Image.open(p) for p in sorted(run.frames_dir.glob("t*.png"))]
    frames[0].save(
        run.gif_path, save_all=True, append_images=frames[1:], duration=1400, loop=0
    )
    return run.gif_path
```

- [ ] **Step 5: Add `replay` to the CLI**

In `cli.py`, inside `_parser()`:

```python
    replay = sub.add_parser("replay", help="re-render a stored run; no network")
    replay.add_argument("run")
```

and add the command function plus its dispatch entry:

```python
def _cmd_replay(args) -> int:
    from .render import render_run

    run = RunDir.open(args.run)
    scenario = load_scenario(run.read_manifest()["scenario_path"])
    print(render_run(scenario, run))
    return 0
```

```python
    return {"run": _cmd_run, "replay": _cmd_replay}[args.command](args)
```

- [ ] **Step 6: Run the tests**

```bash
uv run pytest -v
```

Expected: every test passes, including the two new assertions in `test_events.py` and `test_shift.py`.

- [ ] **Step 7: Commit**

```bash
git add src/jev_costmap tests
git commit -m "$(printf 'feat: side-by-side frames, GIF, and offline replay\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 13: Scoring, the report, `explain` and `disagree`

**Files:**
- Create: `src/jev_costmap/report.py`
- Modify: `src/jev_costmap/cli.py` (add `explain` and `disagree`; call `write_report` at the end of `run` and `replay`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: stored artifacts via `RunDir`; `World`, `Scenario`.
- Produces: `TickScore(tick, source, blocked_violations, avoid_traversals, false_blocks, path_cells, deferrals)`, `score_tick(world, truth, zone_costs, plans) -> TickScore`, `score_run(scenario, run) -> dict[str, list[TickScore]]`, `write_report(scenario, run) -> Path`, `explain_text(scenario, run, zone, tick) -> str`, `disagree_text(scenario, run) -> str`.

Counting rules, fixed so the numbers mean one thing:

- `blocked_violations` counts **distinct (AGV, zone) entries** into a zone whose truth label is `blocked`, not cells. One AGV crossing one blocked zone is one violation however wide the zone is.
- `avoid_traversals` counts the same way for `avoid`.
- `false_blocks` counts zones the model called blocked whose truth is `clear`. It is per zone, not per AGV.
- `path_cells` is the total number of moves across all AGVs, so caution can be priced against distance.
- A deferred AGV contributes no violations and no cells, and one to `deferrals`.
- The **starting zone** is excluded, not merely the starting cell: an AGV already standing in a zone did not choose to enter it this tick, and it is not charged for the cells it drives while leaving.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:

```python
import datetime as dt

from jev_costmap.baseline import DEFAULT_RULES
from jev_costmap.costs import DEFAULT_WEIGHTS, ZoneCost
from jev_costmap.events import load_scenario
from jev_costmap.judge import Judge
from jev_costmap.planner import Plan
from jev_costmap.report import disagree_text, explain_text, score_run, score_tick, write_report
from jev_costmap.runs import RunDir
from jev_costmap.shift import run_shift
from jev_costmap.world import load_world_file
from tests.test_shift import CalmClient


def world():
    return load_world_file("tests/fixtures/mini.yaml")


def costs(blocked=()):
    return {z: ZoneCost(z, 1.0, z in blocked, {}) for z in ("aisle-1", "aisle-2")}


def test_entering_a_blocked_zone_is_one_violation_per_agv():
    plans = {"agv-1": Plan("agv-1", ((0, 5), (0, 4), (0, 2), (1, 2)), 3.0, False)}
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(), plans)
    assert s.blocked_violations == 1
    assert s.path_cells == 3


def test_the_zone_the_agv_started_in_does_not_count_as_an_entry():
    # agv-1 begins inside aisle-1 and drives out of it. It did not choose to be
    # there, so leaving is not a violation.
    plans = {"agv-1": Plan("agv-1", ((0, 1), (1, 1), (1, 4)), 2.0, False)}
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(), plans)
    assert s.blocked_violations == 0
    assert s.path_cells == 2


def test_avoid_is_counted_separately():
    plans = {"agv-1": Plan("agv-1", ((0, 5), (0, 4), (0, 2)), 2.0, False)}
    s = score_tick(world(), {"aisle-1": "avoid", "aisle-2": "clear"}, costs(), plans)
    assert s.avoid_traversals == 1
    assert s.blocked_violations == 0


def test_blocking_a_clear_zone_is_a_false_block():
    s = score_tick(world(), {"aisle-1": "clear", "aisle-2": "clear"}, costs(blocked=("aisle-1",)), {})
    assert s.false_blocks == 1


def test_blocking_a_genuinely_blocked_zone_is_not_a_false_block():
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(blocked=("aisle-1",)), {})
    assert s.false_blocks == 0


def test_a_deferred_agv_contributes_a_deferral_and_nothing_else():
    plans = {"agv-1": Plan("agv-1", ((0, 1),), 0.0, True)}
    s = score_tick(world(), {"aisle-1": "blocked", "aisle-2": "clear"}, costs(), plans)
    assert (s.deferrals, s.path_cells, s.blocked_violations) == (1, 0, 0)


def recorded(tmp_path):
    s = load_scenario("tests/fixtures/mini-scenario.yaml")
    run = RunDir.create(tmp_path, s.name, when=dt.datetime(2026, 9, 21, 14, 2))
    run_shift(s, Judge(client=CalmClient(), cache_path=tmp_path / "c.json"), run,
              DEFAULT_WEIGHTS, DEFAULT_RULES)
    return s, run


def test_score_run_covers_both_sources_and_every_tick(tmp_path):
    s, run = recorded(tmp_path)
    scores = score_run(s, run)
    assert set(scores) == {"jev", "baseline"}
    assert len(scores["jev"]) == 3


def test_the_report_names_both_models_and_the_totals(tmp_path):
    s, run = recorded(tmp_path)
    text = write_report(s, run).read_text()
    assert "jev" in text and "baseline" in text
    assert "blocked violations" in text.lower()


def test_explain_shows_every_probability_and_the_resulting_multiplier(tmp_path):
    s, run = recorded(tmp_path)
    text = explain_text(s, run, "aisle-1", 1)
    assert "people" in text and "damage" in text and "delay" in text
    assert "offlimits" in text
    assert "multiplier" in text
    assert "confidence" in text


def test_explain_rejects_an_unknown_zone(tmp_path):
    s, run = recorded(tmp_path)
    try:
        explain_text(s, run, "nowhere", 1)
    except KeyError:
        return
    raise AssertionError("expected a KeyError for an unknown zone")


def test_disagree_reports_the_spill_the_baseline_will_not_release(tmp_path):
    """At t=2 the mini scenario's spill has been mopped and the truth label is
    back to clear. The calm client leaves aisle-1 open; the baseline still blocks
    on the word. The row must appear and must name the winner."""
    s, run = recorded(tmp_path)
    text = disagree_text(s, run)
    assert "aisle-1" in text
    assert "jev right" in text


def test_a_difference_of_degree_is_not_scored_as_a_win(tmp_path):
    from jev_costmap.report import _verdict

    assert _verdict(False, False, "clear") == "degree only"
    assert _verdict(False, False, "avoid") == "degree only"
    assert _verdict(False, False, "blocked") == "both wrong"
    assert _verdict(True, False, "blocked") == "jev right"
    assert _verdict(False, True, "clear") == "jev right"
    assert _verdict(True, False, "clear") == "baseline right"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_report.py -v
```

Expected: `ModuleNotFoundError: No module named 'jev_costmap.report'`.

- [ ] **Step 3: Write the implementation**

`src/jev_costmap/report.py`:

```python
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
```

- [ ] **Step 4: Wire the CLI**

In `_parser()`:

```python
    explain = sub.add_parser("explain", help="every probability behind one zone's cost")
    explain.add_argument("run")
    explain.add_argument("--zone", required=True)
    explain.add_argument("--tick", type=int, default=0)

    disagree = sub.add_parser("disagree", help="where Jev and the rule baseline diverge")
    disagree.add_argument("run")
```

Add the commands, and call `write_report` at the end of `_cmd_run` and `_cmd_replay`:

```python
def _open(path):
    run = RunDir.open(path)
    return run, load_scenario(run.read_manifest()["scenario_path"])


def _cmd_explain(args) -> int:
    run, scenario = _open(args.run)
    print(explain_text(scenario, run, args.zone, args.tick))
    return 0


def _cmd_disagree(args) -> int:
    run, scenario = _open(args.run)
    print(disagree_text(scenario, run))
    return 0
```

```python
    return {
        "run": _cmd_run,
        "replay": _cmd_replay,
        "explain": _cmd_explain,
        "disagree": _cmd_disagree,
    }[args.command](args)
```

In `_cmd_run`, after `run_shift(...)`, add:

```python
    render_run(scenario, run)
    write_report(scenario, run)
```

and in `_cmd_replay`, after `render_run(...)`, add `write_report(scenario, run)`.

Import `render_run`, `write_report`, `explain_text` and `disagree_text` at the top of the functions that use them to keep `cli` import-light, matching the existing `_cmd_replay` style.

- [ ] **Step 5: Run the full suite**

```bash
uv run pytest -v
```

Expected: all green (13 tests in `test_report.py`).

- [ ] **Step 6: Commit**

```bash
git add src/jev_costmap tests
git commit -m "$(printf 'feat: ground-truth scoring, report, explain and disagree\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 14: The night-shift scenario

**Files:**
- Create: `config/scenarios/warehouse.yaml`, `config/scenarios/night-shift.yaml`
- Test: `tests/test_scenario.py`

**Interfaces:**
- Consumes: `load_scenario`, `plan_single`, `uniform_layer`.
- Produces: the shipped scenario. No new Python interfaces.

The geometry: 80 x 50 cells at 0.5 m. Six vertical aisles between racking, north and south cross corridors split into two junction zones each, two pick bays on the west wall, staging and charging on the east wall. Fourteen zones, no overlaps, every station reachable.

The event log is written so that the two models disagree in **both** directions and the hidden labels settle each one. Three rows matter:

| tick | what happens | baseline | truth |
| --- | --- | --- | --- |
| 3 | the aisle-3 spill is mopped and signed off | still blocked, on the word `spill` | clear |
| 4 | a stock counter steps in and out of the aisle-2 lane | no keyword matches, so `x1` | blocked |
| 7 | a hydraulic leak in aisle 5, uncleaned | blocked, correctly | blocked |

- [ ] **Step 1: Write the world**

`config/scenarios/warehouse.yaml`:

```yaml
width: 80
height: 50
cell_size_m: 0.5

walls:
  - {x: 0,  y: 0,  w: 80, h: 4}    # north wall
  - {x: 0,  y: 46, w: 80, h: 4}    # south wall
  - {x: 8,  y: 8,  w: 8,  h: 34}   # racking between aisles 1 and 2
  - {x: 20, y: 8,  w: 8,  h: 34}   # racking between aisles 2 and 3
  - {x: 32, y: 8,  w: 8,  h: 34}   # racking between aisles 3 and 4
  - {x: 44, y: 8,  w: 8,  h: 34}   # racking between aisles 4 and 5
  - {x: 56, y: 8,  w: 8,  h: 34}   # racking between aisles 5 and 6
  - {x: 0,  y: 22, w: 4,  h: 6}    # wall between the two pick bays
  - {x: 68, y: 22, w: 12, h: 6}    # wall between staging and charging

zones:
  - id: junction-1
    name: North-west cross corridor
    kind: cross corridor
    description: North cross corridor serving aisles 1 to 3, shared with foot traffic from the office door.
    x: 0
    y: 4
    w: 40
    h: 4
  - id: junction-2
    name: North-east cross corridor
    kind: cross corridor
    description: North cross corridor serving aisles 4 to 6 and the goods-in door.
    x: 40
    y: 4
    w: 40
    h: 4
  - id: junction-3
    name: South-west cross corridor
    kind: cross corridor
    description: South cross corridor serving aisles 1 to 3 and the pick bays.
    x: 0
    y: 42
    w: 40
    h: 4
  - id: junction-4
    name: South-east cross corridor
    kind: cross corridor
    description: South cross corridor serving aisles 4 to 6, staging and the charging bank.
    x: 40
    y: 42
    w: 40
    h: 4
  - id: aisle-1
    name: Aisle 1
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic.
    x: 4
    y: 8
    w: 4
    h: 34
  - id: aisle-2
    name: Aisle 2
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic. Low shelves on the west side are picked by hand.
    x: 16
    y: 8
    w: 4
    h: 34
  - id: aisle-3
    name: Aisle 3
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic.
    x: 28
    y: 8
    w: 4
    h: 34
  - id: aisle-4
    name: Aisle 4
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic.
    x: 40
    y: 8
    w: 4
    h: 34
  - id: aisle-5
    name: Aisle 5
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic.
    x: 52
    y: 8
    w: 4
    h: 34
  - id: aisle-6
    name: Aisle 6
    kind: travel lane
    description: Travel lane between pallet racking, rated for AGV traffic. The widest lane on the floor.
    x: 64
    y: 8
    w: 4
    h: 34
  - id: bay-A
    name: Pick bay A
    kind: pick bay
    description: Pick bay on the west wall with a marked AGV drop point and a walkway behind the racks.
    x: 0
    y: 8
    w: 4
    h: 14
  - id: bay-B
    name: Pick bay B
    kind: pick bay
    description: Pick bay on the west wall with a marked AGV drop point and a walkway behind the racks.
    x: 0
    y: 28
    w: 4
    h: 14
  - id: staging-7
    name: Staging 7
    kind: staging floor
    description: Floor staging for goods waiting to be put away. Pallets stand directly on the deck.
    x: 68
    y: 8
    w: 12
    h: 14
  - id: charging
    name: Charging bank
    kind: charging
    description: AGV charging bank. No stock is kept here and no one works in it during a shift.
    x: 68
    y: 28
    w: 12
    h: 14

stations:
  - {id: dock-A, cell: [2, 12]}
  - {id: dock-B, cell: [2, 34]}
  - {id: stage,  cell: [73, 14]}
  - {id: charge, cell: [73, 34]}

agvs:
  - {id: agv-1, priority: 1, start: dock-A, tasks: [stage, dock-A]}
  - {id: agv-2, priority: 2, start: stage,  tasks: [dock-B, stage]}
  - {id: agv-3, priority: 3, start: dock-B, tasks: [charge, dock-B]}
  - {id: agv-4, priority: 4, start: charge, tasks: [dock-A, charge]}
```

- [ ] **Step 2: Write the shift**

`config/scenarios/night-shift.yaml`:

```yaml
name: night-shift
world: warehouse.yaml
cells_per_tick: 12

# `truth` is the hidden ground truth, authored by hand and never sent to the
# model. It does not persist: every tick states every non-clear zone.
ticks:
  - clock: "13:00"

  - clock: "13:10"
    events:
      - zone: aisle-3
        note: "13:08 Pallet wrap spill reported at the north end of aisle 3. Nobody has been out to it yet."
    truth: {aisle-3: blocked}

  - clock: "13:20"
    events:
      - zone: bay-B
        note: "13:18 Two pickers working bay B until 14:00. They are staying between the racks, clear of the AGV travel lane."
    truth: {aisle-3: blocked, bay-B: avoid}

  - clock: "13:30"
    events:
      - zone: aisle-3
        note: "13:28 Spill mopped, floor dried and signed off by the shift lead."
    truth: {bay-B: avoid}

  - clock: "13:40"
    events:
      - zone: aisle-2
        note: "13:38 Stock count under way in aisle 2. The counter is stepping in and out of the lane to read the low shelves."
    truth: {aisle-2: blocked, bay-B: avoid}

  - clock: "13:50"
    events:
      - zone: staging-7
        note: "13:48 Glassware pallets staged on the floor in staging 7. Shrink wrap has split on one of them."
    truth: {aisle-2: blocked, bay-B: avoid, staging-7: avoid}

  - clock: "14:00"
    events:
      - zone: bay-B
        note: "14:00 The pickers have finished in bay B and signed off the floor."
    truth: {aisle-2: blocked, staging-7: avoid}

  - clock: "14:10"
    events:
      - zone: aisle-5
        note: "14:08 Hydraulic fluid leaking from a pallet truck at the north end of aisle 5. Not yet cleaned."
    truth: {aisle-2: blocked, aisle-5: blocked, staging-7: avoid}

  - clock: "14:20"
    events:
      - zone: aisle-2
        note: "14:18 Stock count finished. Aisle 2 is clear and back in service."
    truth: {aisle-5: blocked, staging-7: avoid}

  - clock: "14:30"
    events:
      - zone: junction-4
        note: "14:28 AGV-7 has stalled across the south-east cross corridor and is waiting on a technician."
    truth: {aisle-5: blocked, junction-4: blocked, staging-7: avoid}

  - clock: "14:40"
    events:
      - zone: aisle-5
        note: "14:38 Hydraulic leak cleaned, floor dried, aisle 5 signed back into service."
    truth: {junction-4: blocked, staging-7: avoid}

  - clock: "14:50"
    events:
      - zone: junction-4
        note: "14:48 Technician recovered AGV-7. The south-east corridor is clear."
    truth: {staging-7: avoid}
```

- [ ] **Step 3: Write the test**

`tests/test_scenario.py`:

```python
import pytest

from jev_costmap.baseline import DEFAULT_RULES, baseline_costs
from jev_costmap.events import initial_agvs, load_scenario, state_at
from jev_costmap.planner import plan_single
from tests.helpers import uniform_layer

SCENARIO = "config/scenarios/night-shift.yaml"


def scenario():
    return load_scenario(SCENARIO)


def test_the_floor_has_fourteen_zones_and_no_overlaps():
    assert len(scenario().world.zones) == 14      # load_world raises on overlap


def test_the_shift_runs_twelve_ticks():
    assert scenario().tick_count == 12


def test_every_station_is_reachable_from_every_other():
    s = scenario()
    layer = uniform_layer(s.world)
    cells = [station.cell for station in s.world.stations.values()]
    for start in cells:
        for goal in cells:
            if start == goal:
                continue
            assert plan_single(s.world, layer, "probe", start, goal) is not None


def test_no_agv_starts_on_a_wall():
    s = scenario()
    for agv in initial_agvs(s.world).values():
        assert s.world.passable(agv.cell)


def test_the_baseline_will_not_release_the_mopped_spill():
    """Tick 3 mopped it and the truth says clear. The rule list cannot tell."""
    s = scenario()
    ws = state_at(s, 3, initial_agvs(s.world))
    assert ws.truth["aisle-3"] == "clear"
    assert baseline_costs(s.world, ws, DEFAULT_RULES)["aisle-3"].blocked is True


def test_the_baseline_cannot_see_the_stock_counter_in_the_lane():
    """Tick 4 puts a person in the aisle-2 lane in words no rule matches."""
    s = scenario()
    ws = state_at(s, 4, initial_agvs(s.world))
    assert ws.truth["aisle-2"] == "blocked"
    cost = baseline_costs(s.world, ws, DEFAULT_RULES)["aisle-2"]
    assert cost.blocked is False
    assert cost.multiplier == pytest.approx(1.0)


def test_the_baseline_does_catch_the_uncleaned_leak():
    """The baseline is a fair opponent: tick 7 is a case it gets right."""
    s = scenario()
    ws = state_at(s, 7, initial_agvs(s.world))
    assert ws.truth["aisle-5"] == "blocked"
    assert baseline_costs(s.world, ws, DEFAULT_RULES)["aisle-5"].blocked is True


def test_a_blocked_junction_never_disconnects_the_floor():
    """junction-4 is blocked at ticks 9 and 10; traffic must still route north."""
    s = scenario()
    layer = uniform_layer(s.world, blocked=("junction-4",))
    assert plan_single(s.world, layer, "probe", (2, 12), (73, 34)) is not None
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/test_scenario.py -v
```

Expected: 8 passed. If `test_every_station_is_reachable_from_every_other` fails, the racking rectangles have sealed an aisle; check the wall list against the aisle x positions before changing anything else.

- [ ] **Step 5: Check the state fits Jev's budget**

```bash
uv run jev-costmap run --scenario config/scenarios/night-shift.yaml --dry-run | wc -c
```

Expected: well under 128,000 characters. Jev 1.13 allows 64k tokens for state plus all questions; 14 zones x 4 questions plus a small state is roughly 6k tokens, so there is ample headroom. If this number is anywhere near 128,000, stop and shorten the level texts before running for real.

- [ ] **Step 6: Commit**

```bash
git add config/scenarios tests/test_scenario.py
git commit -m "$(printf 'feat: the night-shift warehouse scenario\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

### Task 15: Live smoke test, the shipped run, and the README

**Files:**
- Create: `tests/test_live.py`, `README.md`, `runs/example-night-shift/` (committed artifacts)
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: everything.
- Produces: no new Python interfaces.

- [ ] **Step 1: Write the live smoke test**

`tests/test_live.py`:

```python
import os

import pytest

from jev_costmap.events import initial_agvs, load_scenario, state_at
from jev_costmap.judge import Judge
from jev_costmap.questions import build_questions
from jev_costmap.state import build_state

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
```

- [ ] **Step 2: Run it**

```bash
uv run pytest tests/test_live.py -v -m live
```

Expected: 1 passed. Without a key: 1 skipped.

- [ ] **Step 3: Record the shipped run**

```bash
uv run jev-costmap run --scenario config/scenarios/night-shift.yaml --out runs
```

Rename the produced directory to `runs/example-night-shift` so the README can point at a stable path, then re-render it in place:

```bash
mv runs/*-night-shift runs/example-night-shift
uv run jev-costmap replay runs/example-night-shift
```

Read `runs/example-night-shift/report.md` before going further.

- [ ] **Step 4: Check the demo actually demonstrates something**

```bash
uv run jev-costmap disagree runs/example-night-shift
uv run jev-costmap explain runs/example-night-shift --zone aisle-3 --tick 3
uv run jev-costmap explain runs/example-night-shift --zone aisle-2 --tick 4
```

What to look for, and what to do if it is not there:

- **Tick 3, aisle-3.** Jev should not block it and the baseline should. If Jev still blocks it, the correction is not landing; check that the notes reach the state in order and that the `offlimits` instructions are the ones in `questions.py`. Report what you see either way — do not tune the weights to force this.
- **Tick 4, aisle-2.** Jev's `people` score should be high and the baseline `x1.0`.
- **Weights.** Only tune `config/weights.yaml` if a zone Jev scored correctly is not actually being avoided by the planner — that is a pricing problem, not a judgment problem. Never tune to change what Jev said.
- **A tick where the baseline wins.** If there is one, it belongs in the README. Do not remove it.

If the run contradicts the plan's expectations, say so in the handoff rather than adjusting the scenario until it agrees. A demo that reports an honest mixed result is worth more than one that was fitted.

- [ ] **Step 5: Write the README**

`README.md`, following jev-cleaner's shape: what it does, the table that makes the case, install, use, how it works, what the baseline gets right, cost. Fill the numbers from the run you just recorded — **do not copy the numbers below, they are the shape, not the answer**:

````markdown
# jev-semantic-cost-map

Path planning where the geometry is code and the judgment is [Jev](https://docs.typesafe.ai).

A warehouse floor plan gives you occupancy. It does not tell you that the spill
in aisle 3 was mopped twenty minutes ago, or that the stock counter in aisle 2 is
stepping in and out of the lane. jev-semantic-cost-map hands each named zone to Jev
as four typed questions, turns the answers into a cost layer, and routes four
AGVs over it with a space-time A\*.

| t | What the shift log says | Keyword rules | Jev | Truth |
| --- | --- | --- | --- | --- |
| 3 | aisle 3: spill mopped, floor dried, signed off | blocked | ... | clear |
| 4 | aisle 2: counter stepping in and out of the lane | x1.0 | ... | blocked |
| 7 | aisle 5: hydraulic fluid leaking, not yet cleaned | blocked | ... | blocked |

Row 7 is the point as much as rows 3 and 4: the rule list is a fair opponent and
it gets the obvious cases right. What it cannot do is read a correction, a time,
or a spatial qualifier.

## What Jev decides, and what it does not

Jev answers four questions per zone: how exposed a person would be, how likely a
pass is to damage something, how much delay to expect, and whether the zone is
closed outright. Everything else is ordinary code — occupancy, collision
checking, search, reservations, cost arithmetic, rendering.

## Install

```bash
uv sync
export TYPESAFE_API_KEY=...   # from https://console.typesafe.ai/
```

## Use

```bash
jev-costmap run --scenario config/scenarios/night-shift.yaml   # plan the shift
jev-costmap run --scenario ... --dry-run     # print the state and questions, no call
jev-costmap replay runs/example-night-shift  # re-render offline, no key needed
jev-costmap explain runs/example-night-shift --zone aisle-3 --tick 3
jev-costmap disagree runs/example-night-shift
```

A twelve-tick shift is about N input tokens, roughly $N at Jev's current price.

## Scoring

Every tick of a scenario carries hidden ground-truth labels, authored by hand and
never sent to the model. Both planners are scored against them, so the report says
which one was right rather than only that the paths differed.

## What it does not do

No policy re-weighting from the CLI, no confidence-gated escalation, no web
viewer, no continuous-space planning, no optimal multi-agent search. Prioritized
planning can starve a low-priority AGV; when it does, the report says so.
````

- [ ] **Step 6: Verify the whole thing from clean**

```bash
uv run pytest -v                                    # every test, no network
uv run pytest -v -m live                            # the one live test
git status --short                                  # nothing unexpected untracked
uv run jev-costmap replay runs/example-night-shift  # works with the key unset
```

Confirm `runs/example-night-shift` is tracked despite `.gitignore` — the negation
`!runs/example-night-shift/` in `.gitignore` covers it; if `git status` disagrees,
add it with `git add -f`.

- [ ] **Step 7: Commit**

```bash
git add README.md tests/test_live.py
git add -f runs/example-night-shift
git commit -m "$(printf 'feat: live smoke test, shipped example run, and README\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>')"
```

---

## Done when

- `uv run pytest` is green with no network access.
- `uv run pytest -m live` passes with a key and skips without one.
- `jev-costmap replay runs/example-night-shift` works with `TYPESAFE_API_KEY` unset.
- `runs/example-night-shift/report.md` scores both models against the hidden labels.
- `jev-costmap explain ... --zone aisle-3 --tick 3` shows every level probability and confidence behind that zone's cost.
- The README's table is filled from the recorded run, including any row the baseline won.
