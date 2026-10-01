from __future__ import annotations

from typing import Any

import payntbind

import paynt.colored_mdp
import paynt.mdp_family._utils
import paynt.mdp_family.task
import paynt.parameter_space.parameter_space
import paynt.model.model
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)


class MdpFamilyColoredMdpFactory:
    feature_kind = "family"

    def __init__(
        self,
        underlying_mdp: Any,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        coloring: Any,
        build_task: paynt.mdp_family.task.MdpFamilyTask,
        use_exact: bool = False,
    ):
        self.build_task = build_task
        self.use_exact = use_exact
        self.memory_unfolder: Any = None

        if self.build_task.memory_size > 1:
            underlying_mdp, parameter_space, coloring = self.unfold_scheduler_memory(underlying_mdp, parameter_space, coloring)

        self.underlying_mdp = underlying_mdp
        self.parameter_space = parameter_space
        self.coloring = coloring

        self.action_labels: list[str]
        self.choice_to_action: list[int]
        self.action_labels, self.choice_to_action = payntbind.synthesis.extractActionLabels(underlying_mdp)
        self.num_actions = len(self.action_labels)
        self.state_action_choices = MdpFamilyColoredMdpFactory.map_state_action_to_choices(underlying_mdp, self.num_actions, self.choice_to_action)
        self.state_to_actions = MdpFamilyColoredMdpFactory.map_state_to_available_actions(self.state_action_choices)

    def build(self) -> paynt.colored_mdp.ColoredMdp:
        """Overridable so subclasses (e.g. PomdpFamilyColoredMdpFactory) can produce their own Info object while reusing all of the construction above.

        Not called automatically -- the caller (e.g. paynt.api.get_synthesizer) requests a ColoredMdp explicitly.
        """
        colored_mdp = paynt.colored_mdp.ColoredMdp(self.underlying_mdp, self.parameter_space, self.coloring, self.use_exact, feature_kind="family")
        colored_mdp.feature_info = paynt.mdp_family._utils.MdpFamilyInfo(
            num_actions=self.num_actions,
            action_labels=self.action_labels,
            choice_to_action=self.choice_to_action,
            state_action_choices=self.state_action_choices,
            state_to_actions=self.state_to_actions,
        )
        return colored_mdp

    def unfold_scheduler_memory(
        self, underlying_mdp: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, coloring: Any
    ) -> tuple[Any, paynt.parameter_space.parameter_space.ParameterSpace, Any]:
        """Unfold the scheduler memory of the underlying MDP to the initial_memory_size.

        :returns: a new underlying MDP with unfolded scheduler memory
        """
        paynt.utils.error_handling.require_pair_list_coloring(coloring, "unfolding scheduler memory")
        logger.info(f"unfolding scheduler memory of {self.build_task.memory_size} into the model.")

        # unfold the scheduler memory into the model
        self.memory_unfolder = payntbind.synthesis.MemoryUnfolder(underlying_mdp)
        unfolded_mdp = self.memory_unfolder.construct_unfolded_model(self.build_task.memory_size)

        # create new coloring
        choice_to_parameter_options = []
        original_choice_to_parameter_options = coloring.getChoiceToAssignment()
        choice_map = list(self.memory_unfolder.choice_map)
        for choice in range(unfolded_mdp.nr_choices):
            original_choice = choice_map[choice]
            choice_to_parameter_options.append(original_choice_to_parameter_options[original_choice])

        new_coloring = payntbind.synthesis.Coloring(parameter_space.native, unfolded_mdp.nondeterministic_choice_indices, choice_to_parameter_options)

        logger.info(f"unfolded model has {unfolded_mdp.nr_states} states and {unfolded_mdp.nr_choices} choices.")

        return unfolded_mdp, parameter_space, new_coloring

    @staticmethod
    def map_state_action_to_choices(mdp: Any, num_actions: int, choice_to_action: list[int]) -> list[list[list[int]]]:
        state_action_choices = []
        for state in range(mdp.nr_states):
            action_choices: list[list[int]] = [[] for action in range(num_actions)]
            for choice in mdp.transition_matrix.get_rows_for_group(state):
                action = choice_to_action[choice]
                action_choices[action].append(choice)
            state_action_choices.append(action_choices)
        return state_action_choices

    @staticmethod
    def map_state_to_available_actions(state_action_choices: list[list[list[int]]]) -> list[list[int]]:
        state_to_actions = []
        for _state, action_choices in enumerate(state_action_choices):
            available_actions = []
            for action, choices in enumerate(action_choices):
                if choices:
                    available_actions.append(action)
            state_to_actions.append(available_actions)
        return state_to_actions
