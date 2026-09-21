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


def test_matching_is_case_insensitive():
    """Keyword appears only in uppercase; match requires .lower()."""
    costs = baseline_costs(world(), state({"aisle-1": ("PICKERS working in BAY B",)}), (Rule("picker", 3.0, False),))
    assert costs["aisle-1"].multiplier == pytest.approx(3.0)


def test_matching_reads_the_description_too():
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
