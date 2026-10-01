"""Internal splitting-heuristic scoring, shared by every search algorithm that needs to estimate which parameter to blame for an inconsistent scheduler."""

from __future__ import annotations

from typing import Any

import math

import payntbind

import paynt.colored_mdp
import paynt.utils.error_handling


def estimate_scheduler_difference(
    colored_mdp: paynt.colored_mdp.ColoredMdp,
    mdp: Any,
    underlying_mdp_choice_map: list[int],
    inconsistent_assignments: dict[int, list[int]],
    choice_values: list[float],
    expected_visits: list[float],
) -> dict[int, float]:
    """Default AR-splitting heuristic: estimate, per inconsistent parameter, how much the choice values differ across the parameter's options (weighted by
    expected visits)."""
    paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, "AR split scoring (estimate_scheduler_difference)")
    return payntbind.synthesis.computeInconsistentParameterVariance(
        colored_mdp.parameter_space.native,
        mdp.nondeterministic_choice_indices,
        underlying_mdp_choice_map,
        choice_values,
        colored_mdp.coloring,
        inconsistent_assignments,
        expected_visits,
    )


def estimate_scheduler_difference_pomdp(
    colored_mdp: paynt.colored_mdp.ColoredMdp,
    mdp: Any,
    underlying_mdp_choice_map: list[int],
    inconsistent_assignments: dict[int, list[int]],
    choice_values: list[float],
    expected_visits: list[float],
) -> dict[int, float]:
    """POMDP specialized variant of estimate_scheduler_difference, hand-optimized for posterior-unaware unfolding using
    colored_mdp.feature_info.parameter_option_to_actions (the reverse coloring built during unfolding) instead of the generic payntbind call."""
    # create inverse underlying-choice-to-restricted-choice map
    # TODO optimize this for multiple properties
    underlying_to_restricted_action_map: list[int | None] = [None] * colored_mdp.underlying_mdp.nr_choices
    for choice in range(mdp.nr_choices):
        underlying_to_restricted_action_map[underlying_mdp_choice_map[choice]] = choice

    # map choices to their origin states
    choice_to_state = []
    tm = mdp.transition_matrix
    for state in range(mdp.nr_states):
        for _choice in tm.get_rows_for_group(state):
            choice_to_state.append(state)

    # for each parameter, compute its difference sum and a number of affected states
    inconsistent_differences: dict[int, float] = {}
    for parameter_index, options in inconsistent_assignments.items():
        difference_sum = 0.0
        states_affected = 0
        edges_0 = colored_mdp.feature_info.parameter_option_to_actions[parameter_index][options[0]]
        for choice_index, _ in enumerate(edges_0):
            choice_0_global = edges_0[choice_index]
            choice_0 = underlying_to_restricted_action_map[choice_0_global]
            if choice_0 is None:
                continue

            source_state = choice_to_state[choice_0]
            source_state_visits = expected_visits[source_state]

            if source_state_visits == 0:
                continue

            state_values = []
            for option in options:
                assert len(colored_mdp.feature_info.parameter_option_to_actions[parameter_index][option]) > choice_index
                choice_global = colored_mdp.feature_info.parameter_option_to_actions[parameter_index][option][choice_index]
                choice = underlying_to_restricted_action_map[choice_global]
                choice_value = choice_values[choice]
                state_values.append(choice_value)

            min_value = min(state_values)
            max_value = max(state_values)
            difference = (max_value - min_value) * source_state_visits
            assert not math.isnan(difference)
            difference_sum += difference
            states_affected += 1

        if states_affected == 0:
            parameter_score = 0.0
        else:
            parameter_score = difference_sum / states_affected
        inconsistent_differences[parameter_index] = parameter_score

    return inconsistent_differences


def parameters_with_max_score(parameter_score: dict[int, float]) -> list[int]:
    max_score = max(parameter_score.values())
    return [parameter_index for parameter_index in parameter_score if parameter_score[parameter_index] == max_score]


def compute_incompatibility_levels(candidate_selections: list[list[list[int]]]) -> dict[int, list[int]]:
    """
    Paper https://www.jair.org/index.php/jair/article/view/16593 Section 3.3's L(h): given the
    primary_selection of every still-undecided property whose OWN scheduler is already fully consistent (a
    genuine candidate delta_i in Delta), find parameters where those candidates disagree. Returns
    {parameter: distinct_values} only where L(h) > 1 (>=2 distinct values);
    empty if fewer than 2 candidates given. Order of each returned value list is
    first-candidate-seen order, not sorted -- deterministic, and critical for the N=1 byte-identical guarantee
    at the call site (see synthesizer_ar.split_parameter_space).
    """
    if len(candidate_selections) < 2:
        return {}
    disagreements: dict[int, list[int]] = {}
    for parameter in range(len(candidate_selections[0])):
        values: list[int] = []
        for selection in candidate_selections:
            options = selection[parameter]
            if len(options) == 1 and options[0] not in values:
                values.append(options[0])
        if len(values) > 1:
            disagreements[parameter] = values
    return disagreements


def parameters_with_max_incompatibility(disagreement_levels: dict[int, list[int]]) -> list[int]:
    """Argmax L(h); same tie-all-winners convention as parameters_with_max_score."""
    if not disagreement_levels:
        return []
    max_level = max(len(values) for values in disagreement_levels.values())
    return [parameter for parameter, values in disagreement_levels.items() if len(values) == max_level]
