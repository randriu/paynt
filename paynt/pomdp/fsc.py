from __future__ import annotations

from typing import Any

import json


class Fsc:
    """A finite-state controller.

    An FSC having
    - a fixed number of nodes
    - a joint transition function of the type NxZ -> Distr(ActxN), where f(n,z) is a dictionary
        (action,memory) -> probability
    """

    def __init__(self, num_nodes: int, num_observations: int):
        self.num_nodes = num_nodes
        self.num_observations = num_observations
        self.transitions: list[list[Any]] = [[None] * num_observations for _ in range(num_nodes)]
        self.observation_labels: list[str] | None = None
        self.action_labels: list[str] | None = None

    def check_transitions(self, observation_to_actions: list[list[int]]) -> None:
        assert len(self.transitions) == self.num_nodes, "FSC transition_function function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.transitions[node]) == self.num_observations, f"in memory node {node}, FSC transition function is not defined for all observations"
            for obs in range(self.num_observations):
                for action, update in self.transitions[node][obs].keys():
                    assert action in observation_to_actions[obs], f"in observation {obs} FSC chooses invalid action {action}"
                    assert 0 <= update and update < self.num_nodes, f"invalid FSC memory update {update}"

    def check(self, observation_to_actions: list[list[int]]) -> None:
        """Check whether fields of FSC have been initialized appropriately."""
        assert self.num_nodes > 0, "FSC must have at least 1 node"
        self.check_transitions(observation_to_actions)


class FscFactored:
    """A finite-state controller encoded in factored form.

    Class for encoding an FSC having
    - a fixed number of nodes
    - action selection is either:
        + deterministic: gamma: NxZ -> Act, or
        + randomized: gamma: NxZ -> Distr(Act), where gamma(n,z) is a dictionary of pairs (action,probability)
    - deterministic posterior-unaware memory update delta: NxZ -> N
    """

    def __init__(self, num_nodes: int, num_observations: int, is_deterministic: bool = False):
        self.num_nodes = num_nodes
        self.num_observations = num_observations
        self.is_deterministic = is_deterministic

        # each cell is an action index (int) when is_deterministic, or a dict[int,float] (action -> probability)
        # otherwise -- kept as Any rather than a union, since which shape applies is a whole-FSC-wide
        # invariant (self.is_deterministic) rather than something checked cell-by-cell
        self.action_function: list[list[Any]] = [[None] * num_observations for _ in range(num_nodes)]
        # similarly: an int (memory index) when deterministic, or a dict[int,float] otherwise
        self.update_function: list[list[Any]] = [[None] * num_observations for _ in range(num_nodes)]

        self.observation_labels: list[str] | None = None
        self.action_labels: list[str] | None = None

    def __str__(self) -> str:
        return json.dumps(self.to_json(), indent=4)

    def action_function_signature(self) -> str:
        if self.is_deterministic:
            return " NxZ -> Act"
        return " NxZ -> Distr(Act)"

    def to_json(self) -> dict[str, Any]:
        json_dict: dict[str, Any] = {}
        json_dict["num_nodes"] = self.num_nodes
        json_dict["num_observations"] = self.num_observations
        json_dict["__comment_action_function"] = f"action function has signature {self.action_function_signature()}"
        json_dict["__comment_update_function"] = "update function has signature NxZ -> N"

        json_dict["action_function"] = self.action_function
        json_dict["update_function"] = self.update_function

        if self.action_labels is not None:
            json_dict["action_labels"] = self.action_labels
        if self.observation_labels is not None:
            json_dict["observation_labels"] = self.observation_labels

        return json_dict

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> FscFactored:
        """The FSC of to_json: a stochastic one has distributions (dicts) in its cells, whose keys JSON made strings."""
        cells = [cell for function in ("action_function", "update_function") for row in json_dict[function] for cell in row]
        is_deterministic = not any(isinstance(cell, dict) for cell in cells)
        fsc = FscFactored(json_dict["num_nodes"], json_dict["num_observations"], is_deterministic)
        if is_deterministic:
            fsc.action_function = json_dict["action_function"]
            fsc.update_function = json_dict["update_function"]
        else:
            fsc.action_function = [[_distribution_from_json(cell) for cell in row] for row in json_dict["action_function"]]
            fsc.update_function = [[_distribution_from_json(cell) for cell in row] for row in json_dict["update_function"]]
        fsc.action_labels = json_dict.get("action_labels")
        fsc.observation_labels = json_dict.get("observation_labels")
        return fsc

    def reorder_nodes(self, node_old_to_new: list[int]) -> None:
        def renamed(update: Any) -> Any:
            if update is None:
                return None
            if self.is_deterministic:
                return node_old_to_new[update]
            return {node_old_to_new[node]: probability for node, probability in update.items()}

        action_function: list[Any] = [None for node in range(self.num_nodes)]
        update_function: list[Any] = [None for node in range(self.num_nodes)]
        for node_old, node_new in enumerate(node_old_to_new):
            action_function[node_new] = self.action_function[node_old]
            update_function[node_new] = [renamed(update) for update in self.update_function[node_old]]
        self.action_function = action_function
        self.update_function = update_function

    def reorder_actions(self, action_labels: list[str]) -> None:
        assert self.action_labels is not None
        for node in range(self.num_nodes):
            for obs in range(self.num_observations):
                if self.is_deterministic:
                    action = self.action_function[node][obs]
                    self.action_function[node][obs] = action_labels.index(self.action_labels[action])
                else:
                    action_function = {}
                    for action, prob in self.action_function[node][obs].items():
                        action_function[action_labels.index(self.action_labels[action])] = prob
                    self.action_function[node][obs] = action_function
        self.action_labels = action_labels.copy()

    def make_stochastic(self) -> None:
        if not self.is_deterministic:
            return
        for node in range(self.num_nodes):
            for obs in range(self.num_observations):
                self.action_function[node][obs] = {self.action_function[node][obs]: 1.0}
                self.update_function[node][obs] = {self.update_function[node][obs]: 1.0}
        self.is_deterministic = False

    def check_action_function(self, observation_to_actions: list[list[int]]) -> None:
        assert len(self.action_function) == self.num_nodes, "FSC action function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.action_function[node]) == self.num_observations, f"in memory node {node}, FSC action function is not defined for all observations"
            for obs in range(self.num_observations):
                if observation_to_actions[obs] == []:
                    assert self.action_function[node][obs] is None
                    continue
                if self.is_deterministic:
                    action_support = [self.action_function[node][obs]]
                else:
                    action_support = self.action_function[node][obs].keys()
                for action in action_support:
                    assert action in observation_to_actions[obs], f"in observation {obs} FSC chooses invalid action {action}"

    def check_update_function(self) -> None:
        assert len(self.update_function) == self.num_nodes, "FSC update function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.update_function[node]) == self.num_observations, f"in memory node {node}, FSC update function is not defined for all observations"
            for obs in range(self.num_observations):
                update = self.update_function[node][obs]
                continue  # skipping check due to inconsistency in our FSC definition
                assert 0 <= update and update < self.num_nodes, f"invalid FSC memory update {update}"

    def check(self, observation_to_actions: list[list[int]]) -> None:
        """Check whether fields of FSC have been initialized appropriately."""
        assert self.num_nodes > 0, "FSC must have at least 1 node"
        self.check_action_function(observation_to_actions)
        self.check_update_function()

    def fill_trivial_actions(self, observation_to_actions: list[list[int]]) -> None:
        """For each observation with 1 available action, set gamma(n,z) to that action."""
        for obs, actions in enumerate(observation_to_actions):
            if len(actions) != 1:
                continue
            action: Any = actions[0]
            if not self.is_deterministic:
                action = {action: 1}
            for node in range(self.num_nodes):
                self.action_function[node][obs] = action

    def fill_trivial_updates(self, observation_to_actions: list[list[int]]) -> None:
        """For each observation with 1 available action, set delta(n,z) to n."""
        for obs, actions in enumerate(observation_to_actions):
            if len(actions) > 1:
                continue
            for node in range(self.num_nodes):
                self.update_function[node][obs] = node

    def fill_zero_updates(self) -> None:
        for node in range(self.num_nodes):
            self.update_function[node] = [0] * self.num_observations

    def fill_implicit_actions_and_updates(self) -> None:
        """For an FSC with an irregular memory model, make action and updates explicit."""
        for node in range(self.num_nodes):
            for obs in range(self.num_observations):
                if self.action_function[node][obs] is None:
                    self.action_function[node][obs] = self.action_function[0][obs]
                    self.update_function[node][obs] = self.update_function[0][obs]

    def copy(self) -> FscFactored:
        fsc = FscFactored(self.num_nodes, self.num_observations, self.is_deterministic)
        fsc.action_function = [list(node) for node in self.action_function]
        fsc.update_function = [list(node) for node in self.update_function]
        fsc.observation_labels = self.observation_labels
        fsc.action_labels = self.action_labels
        return fsc


def _distribution_from_json(cell: dict[str, float] | None) -> dict[int, float] | None:
    return None if cell is None else {int(key): probability for key, probability in cell.items()}
