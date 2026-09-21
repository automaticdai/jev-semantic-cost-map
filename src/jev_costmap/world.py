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
