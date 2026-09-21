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

    cmap = matplotlib.colormaps["YlOrRd"].with_extremes(bad="#d9d9d9")
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
    try:
        frames[0].save(
            run.gif_path, save_all=True, append_images=frames[1:], duration=1400, loop=0
        )
    finally:
        for frame in frames:
            frame.close()
    return run.gif_path
