from __future__ import annotations

from typing import Any

import stormpy

import paynt.specification.property
import paynt.specification.property_result
import paynt.utils.error_handling

import payntbind

import math

import logging

logger = logging.getLogger(__name__)


class Mdp:
    @classmethod
    def assert_no_overlapping_guards(cls, model: Any) -> None:
        if model.labeling.contains_label("overlap_guards"):
            assert model.labeling.get_states("overlap_guards").number_of_set_bits() == 0

    def __init__(self, model: Any):
        # Mdp.assert_no_overlapping_guards(model)
        self.model = model
        if len(model.initial_states) > 1:
            logger.warning("WARNING: obtained model with multiple initial states")

    @property
    def states(self) -> int:
        return self.model.nr_states

    @property
    def is_deterministic(self) -> bool:
        return self.model.nr_choices == self.model.nr_states

    @property
    def initial_state(self) -> int:
        return self.model.initial_states[0]

    def model_check_property(self, prop: paynt.specification.property.Property, alt: bool = False) -> paynt.specification.property_result.PropertyResult:
        formula = prop.formula if not alt else prop.formula_alt
        result = paynt.specification.property.Property.model_check(self.model, formula)
        value = result.at(self.initial_state)
        return paynt.specification.property_result.PropertyResult(prop, result, value)

    def check_specification(
        self, spec: paynt.specification.property.Specification, constraint_indices: Any = None, short_evaluation: bool = False
    ) -> paynt.specification.property_result.SpecificationResult:
        """Check the specification on this model.

        A model that still has nondeterminism (the full assignment of an incomplete coloring or of an MDP family, see ColoredMdp.build_assignment) is checked
        property by property, each in the direction that satisfies it: the remaining choices are resolved in the best way for that property. Several properties
        would each be given a policy of their own instead of a common one, so only a single property is supported for such a model.
        """
        paynt.utils.error_handling.require_single_property_on_nondeterministic_model(self, spec)
        if constraint_indices is None:
            constraint_indices = spec.all_constraint_indices()
        results: list[paynt.specification.property_result.PropertyResult | None] = [None for _ in spec.constraints]
        for index in constraint_indices:
            result = self.model_check_property(spec.constraints[index])
            results[index] = result
            if short_evaluation and result.sat is False:
                break
        spec_result = paynt.specification.property_result.SpecificationResult()
        spec_result.constraints_result = paynt.specification.property_result.ConstraintsResult(results)

        if spec.has_optimality and not (short_evaluation and spec_result.constraints_result.sat is False):
            assert spec.optimality is not None
            spec_result.optimality_result = self.model_check_property(spec.optimality)
        return spec_result


class SubMdp(Mdp):
    def __init__(self, model: Any, underlying_mdp_state_map: list[int], underlying_mdp_choice_map: list[int]):
        super().__init__(model)
        self.underlying_mdp_choice_map = underlying_mdp_choice_map
        self.underlying_mdp_state_map = underlying_mdp_state_map


class Smg(Mdp):
    def __init__(self, model: Any):
        super().__init__(model)

    def model_check_property(self, prop: paynt.specification.property.Property, alt: bool = False) -> paynt.specification.property_result.PropertyResult:
        formula = prop.game_formula if not alt else prop.game_formula_alt

        result = payntbind.synthesis.model_check_smg(
            self.model, formula, only_initial_states=False, set_produce_schedulers=True, env=paynt.specification.property.Property.environment
        )

        value = result.at(self.model.initial_states[0])
        return paynt.specification.property_result.PropertyResult(prop, result, value)


class SubmodelBuilder:
    """Stateless helpers for restricting an MDP to a subset of choices.

    Used by ColoredMdp.restrict()/.build()/ .build_assignment() to construct the induced sub-MDP (or DTMC) for a (sub)parameter space, without any
    parameter/coloring knowledge of its own.
    """

    @staticmethod
    def default_builder_options() -> Any:
        builder_options = stormpy.SubsystemBuilderOptions()
        builder_options.build_state_mapping = True
        builder_options.build_action_mapping = True
        return builder_options

    @staticmethod
    def restrict(mdp: Any, choices: Any, builder_options: Any) -> tuple[Any, list[int], list[int]]:
        """Restrict the MDP to the selected actions.

        :param choices: a bitvector of selected actions
        :return: (1) the restricted model
        :return: (2) sub- to full state mapping
        :return: (3) sub- to full action mapping
        """
        keep_unreachable_states = False  # TODO investigate this
        all_states = stormpy.BitVector(mdp.nr_states, True)
        submodel_construction = stormpy.construct_submodel(mdp, all_states, choices, keep_unreachable_states, builder_options)
        model = submodel_construction.model
        state_map = submodel_construction.new_to_old_state_mapping.copy()
        choice_map = submodel_construction.new_to_old_action_mapping.copy()
        return model, state_map, choice_map

    @staticmethod
    def build_submdp(mdp: Any, choices: Any, builder_options: Any) -> SubMdp:
        model, state_map, choice_map = SubmodelBuilder.restrict(mdp, choices, builder_options)
        return SubMdp(model, state_map, choice_map)

    @staticmethod
    def mdp_to_dtmc(mdp: Any) -> Any:
        tm = mdp.transition_matrix
        tm.make_row_grouping_trivial()
        assert tm.nr_columns == tm.nr_rows, "expected transition matrix without non-trivial row groups"
        if mdp.is_exact:
            components = stormpy.storage.SparseExactModelComponents(tm, mdp.labeling, mdp.reward_models)
            return stormpy.storage.SparseExactDtmc(components)
        components = stormpy.storage.SparseModelComponents(tm, mdp.labeling, mdp.reward_models)
        return stormpy.storage.SparseDtmc(components)


class ModelIndex:
    """Stateless helpers for converting between scheduler/choice representations over a model's index space, for generic graph queries on a stormpy model, and
    for numeric per-choice/per-state facts (choice values, expected visits) derived from a model-checking result.

    Used by ColoredMdp class and by Colored MDP factories (e.g. paynt.dt) that need this plumbing with no parameter/parameter-space/coloring knowledge of their
    own.
    """

    @staticmethod
    def compute_choice_destinations(model: Any, use_exact: bool = False) -> Any:
        if use_exact:
            return payntbind.synthesis.computeChoiceDestinationsExact(model)
        return payntbind.synthesis.computeChoiceDestinations(model)

    @staticmethod
    def empty_scheduler(model: Any) -> list[int | None]:
        return [None] * model.nr_states

    @staticmethod
    def discard_unreachable_choices(model: Any, choice_destinations: Any, state_to_choice: list[int | None]) -> list[int | None]:
        state_to_choice_reachable = ModelIndex.empty_scheduler(model)
        state_visited = [False] * model.nr_states
        initial_state = list(model.initial_states)[0]
        state_visited[initial_state] = True
        state_queue = [initial_state]
        while state_queue:
            state = state_queue.pop()
            choice = state_to_choice[state]
            state_to_choice_reachable[state] = choice
            for dst in choice_destinations[choice]:
                if not state_visited[dst]:
                    state_visited[dst] = True
                    state_queue.append(dst)
        return state_to_choice_reachable

    @staticmethod
    def scheduler_to_state_to_choice(
        underlying_mdp: Any, choice_destinations: Any, submdp: SubMdp, scheduler: Any, discard_unreachable_choices: bool = True
    ) -> list[int | None]:
        """Convert a scheduler over a sub-MDP to a state-to-choice mapping over the underlying MDP.

        param: underlying_mdp: the underlying MDP used to construct the sub-MDP
        param: choice_destinations: the choice destinations of the underlying MDP
        param: submdp: the sub-MDP over which the scheduler is defined
        param: scheduler: a memoryless deterministic scheduler over the sub-MDP
        param: discard_unreachable_choices: if True, unreachable choices in the underlying MDP are discarded
            from the state-to-choice mapping
        return: a state-to-choice mapping over the underlying MDP, where each state is mapped to the choice
            selected by the scheduler in the sub-MDP, or None if the state is not reachable under the scheduler
        """
        if submdp.model.is_exact:
            state_to_underlying_mdp_choice = payntbind.synthesis.schedulerToStateToGlobalChoiceExact(scheduler, submdp.model, submdp.underlying_mdp_choice_map)
        else:
            state_to_underlying_mdp_choice = payntbind.synthesis.schedulerToStateToGlobalChoice(scheduler, submdp.model, submdp.underlying_mdp_choice_map)
        state_to_choice = ModelIndex.empty_scheduler(underlying_mdp)
        for state in range(submdp.model.nr_states):
            underlying_mdp_choice = state_to_underlying_mdp_choice[state]
            underlying_mdp_state = submdp.underlying_mdp_state_map[state]
            state_to_choice[underlying_mdp_state] = underlying_mdp_choice
        if discard_unreachable_choices:
            state_to_choice = ModelIndex.discard_unreachable_choices(underlying_mdp, choice_destinations, state_to_choice)
        return state_to_choice

    @staticmethod
    def state_to_choice_to_choices(model: Any, state_to_choice: list[int | None]) -> Any:
        num_choices = model.nr_choices
        choices = stormpy.BitVector(num_choices, False)
        for choice in state_to_choice:
            if choice is not None and choice < num_choices:
                choices.set(choice, True)
        return choices

    @staticmethod
    def identify_absorbing_states(model: Any) -> list[bool]:
        state_is_absorbing = [True] * model.nr_states
        tm = model.transition_matrix
        for state in range(model.nr_states):
            for choice in tm.get_rows_for_group(state):
                for entry in tm.get_row(choice):
                    if entry.column != state:
                        state_is_absorbing[state] = False
                        break
                if not state_is_absorbing[state]:
                    break
        return state_is_absorbing

    @staticmethod
    def identify_states_with_actions(model: Any) -> list[bool]:
        """Get a mask of states having more than one action."""
        state_has_actions: list[bool] = [False] * model.nr_states
        ndi = model.nondeterministic_choice_indices
        for state in range(model.nr_states):
            num_actions = ndi[state + 1] - ndi[state]
            state_has_actions[state] = num_actions > 1
        return state_has_actions

    @staticmethod
    def identify_target_states(model: Any, prop: paynt.specification.property.Property) -> Any:
        if prop.is_discounted_reward:
            return stormpy.BitVector(model.nr_states, False)
        target_label = prop.get_target_label()
        return model.labeling.get_states(target_label)

    @staticmethod
    def make_vector_defined(vector: list[float]) -> list[float]:
        vector_noinf = [value if value != math.inf else 0 for value in vector]
        default_value = sum(vector_noinf) / len(vector)
        return [value if value != math.inf else default_value for value in vector]

    @staticmethod
    def choice_values(mdp: Any, prop: paynt.specification.property.Property, state_values: list[float]) -> list[float]:
        """Get choice values after model checking MDP against a property.

        Value of choice c: s -> s' is computed as
        rew(c) + sum_s' [ P(s,c,s') * mc(s') ], where
        - rew(c) is the reward associated with choice (c)
        - P(s,c,s') is the probability of transitioning from s to s' under action c
        - mc(s') is the model checking result in state s'
        """
        # multiply probability with model checking results
        if mdp.is_exact:
            choice_values = payntbind.synthesis.multiply_with_vector_exact(mdp.transition_matrix, state_values)
        else:
            choice_values = payntbind.synthesis.multiply_with_vector(mdp.transition_matrix, state_values)
        choice_values = ModelIndex.make_vector_defined(choice_values)

        # if the associated reward model has state-action rewards, then these must be added to choice values
        if prop.reward:
            reward_name = prop.formula.reward_name
            rm = mdp.reward_models.get(reward_name)
            assert rm.has_state_action_rewards
            choice_rewards = list(rm.state_action_rewards)
            assert mdp.nr_choices == len(choice_rewards)
            for choice in range(mdp.nr_choices):
                choice_values[choice] += choice_rewards[choice]

        return choice_values

    @staticmethod
    def compute_expected_visits(mdp: Any, prop: paynt.specification.property.Property, choices: Any, disable_expected_visits: bool = False) -> list[float]:
        """Compute the expected number of visits in the states of the DTMC induced by the given choices."""
        if disable_expected_visits:
            return [1] * mdp.nr_states

        # extract DTMC induced by this MDP-scheduler
        builder_options = SubmodelBuilder.default_builder_options()
        sub_mdp, state_map, _ = SubmodelBuilder.restrict(mdp, choices, builder_options)
        dtmc = SubmodelBuilder.mdp_to_dtmc(sub_mdp)
        dtmc_visits = paynt.specification.property.Property.compute_expected_visits(dtmc)

        # handle infinity- and zero-visits
        if prop.minimizing:
            dtmc_visits = ModelIndex.make_vector_defined(dtmc_visits)
        else:
            dtmc_visits = [value if value != math.inf else 0 for value in dtmc_visits]

        # map vector of expected visits onto the state space of the given mdp
        expected_visits: list[float] = [0] * mdp.nr_states
        for state in range(dtmc.nr_states):
            mdp_state = state_map[state]
            visits = dtmc_visits[state]
            expected_visits[mdp_state] = visits

        return expected_visits
