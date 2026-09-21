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
