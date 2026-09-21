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
