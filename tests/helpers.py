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
    # Multipliers are expected to be >= 1.0, as production `fuse()` guarantees
    # (BLOCK_THRESHOLD aside). A sub-1.0 multiplier here would make a move
    # cost less than the Manhattan heuristic's per-step unit, which would
    # break A*'s admissibility in plan_single.
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
