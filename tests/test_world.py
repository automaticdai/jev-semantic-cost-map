import numpy as np
import pytest
import yaml

from jev_planner.world import load_world


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
