"""FscFactored: renaming its nodes and actions, and filling in the updates of observations with a single action."""

import copy
import json

import pytest

import paynt.pomdp.fsc

PERMUTATION = [1, 2, 0]


def fsc(deterministic):
    """Three nodes, two observations, actions a, b, c."""
    controller = paynt.pomdp.fsc.FscFactored(3, 2, is_deterministic=True)
    controller.action_function = [[0, 1], [2, 0], [1, 1]]
    controller.update_function = [[1, 2], [0, 0], [2, 1]]
    controller.action_labels = ["a", "b", "c"]
    if not deterministic:
        controller.make_stochastic()
    return controller


def as_distribution(value):
    return value if isinstance(value, dict) else {value: 1}


@pytest.mark.parametrize("deterministic", [True, False])
def test_reordering_the_nodes_renames_them(deterministic):
    controller = fsc(deterministic)
    original = copy.deepcopy(controller)
    controller.reorder_nodes(PERMUTATION)
    for node in range(3):
        for obs in range(2):
            renamed = PERMUTATION[node]
            assert controller.action_function[renamed][obs] == original.action_function[node][obs]
            expected = {PERMUTATION[update]: probability for update, probability in as_distribution(original.update_function[node][obs]).items()}
            assert as_distribution(controller.update_function[renamed][obs]) == expected


@pytest.mark.parametrize("deterministic", [True, False])
def test_reordering_the_actions_keeps_what_is_played(deterministic):
    controller = fsc(deterministic)
    original = copy.deepcopy(controller)
    controller.reorder_actions(["c", "a", "b"])
    for node in range(3):
        for obs in range(2):
            played = {controller.action_labels[action]: p for action, p in as_distribution(controller.action_function[node][obs]).items()}
            assert played == {original.action_labels[action]: p for action, p in as_distribution(original.action_function[node][obs]).items()}


def test_reordering_the_nodes_keeps_an_observation_without_an_update():
    controller = paynt.pomdp.fsc.FscFactored(2, 2, is_deterministic=True)
    controller.update_function = [[1, None], [0, None]]
    controller.reorder_nodes([1, 0])
    assert controller.update_function == [[1, None], [0, None]]


@pytest.mark.parametrize("deterministic", [True, False])
def test_an_fsc_read_back_from_json_is_the_same(deterministic):
    controller = fsc(deterministic)
    controller.observation_labels = ["x", "y"]
    written = json.loads(json.dumps(controller.to_json()))
    read_back = paynt.pomdp.fsc.FscFactored.from_json(written)
    assert read_back.is_deterministic == deterministic
    assert (read_back.action_function, read_back.update_function) == (controller.action_function, controller.update_function)
    assert (read_back.action_labels, read_back.observation_labels) == (["a", "b", "c"], ["x", "y"])


def test_an_observation_with_a_single_action_keeps_the_node():
    controller = paynt.pomdp.fsc.FscFactored(2, 2)
    controller.fill_trivial_updates([[0], [0, 1]])
    assert controller.update_function == [[0, None], [1, None]]
