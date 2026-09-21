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
