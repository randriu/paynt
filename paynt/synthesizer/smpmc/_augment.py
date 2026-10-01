"""Turn a family ColoredMdp (feature_kind "family": the sketch's parameters pick the environment, the agent's policy stays nondeterministic) into a generic one
where the policy is explicit parameters too -- so robust synthesis can quantify existentially over the policy and universally over the environment.

Ported from molehill's run() (https://github.com/linusheck/molehill, GPL-3.0): one policy parameter per state
with more than one available action, whose options are that state's actions.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import payntbind

import paynt.colored_mdp
import paynt.utils.error_handling


def _state_names(mdp: Any) -> list[str]:
    if not mdp.has_state_valuations():
        return [f"s{state}" for state in range(mdp.nr_states)]
    names = []
    for state in range(mdp.nr_states):
        valuation = json.loads(str(mdp.state_valuations.get_json(state)))
        # int(): print booleans as 0/1, matching molehill's naming
        names.append("&".join(f"{var}={int(value)}" for var, value in valuation.items() if not var.startswith("_loc_prism2jani")))
    return names


def _shared_action(colored_mdp: paynt.colored_mdp.ColoredMdp, choice_to_assignment: list[list[tuple[int, int]]]) -> tuple[int, int] | None:
    """A (state, action) where some environment enables several choices of that action, or None.

    Choices colored over the same parameters exclude each other unless colored identically; choices colored over different parameters can be enabled together
    iff they agree on the parameters they share.
    """
    for state, action_choices in enumerate(colored_mdp.feature_info.state_action_choices):
        for action, choices in enumerate(action_choices):
            # parameters a choice is colored over -> the colorings of those choices
            colorings: dict[frozenset[int], set[tuple[tuple[int, int], ...]]] = {}
            for choice in choices:
                coloring = tuple(sorted(choice_to_assignment[choice]))
                same_parameters = colorings.setdefault(frozenset(parameter for parameter, _ in coloring), set())
                if coloring in same_parameters:
                    return state, action
                same_parameters.add(coloring)
            for (parameters1, colorings1), (parameters2, colorings2) in itertools.combinations(colorings.items(), 2):
                shared = parameters1 & parameters2
                projected1 = {tuple(pair for pair in coloring if pair[0] in shared) for coloring in colorings1}
                if any(tuple(pair for pair in coloring if pair[0] in shared) in projected1 for coloring in colorings2):
                    return state, action
    return None


def _partial_action(colored_mdp: paynt.colored_mdp.ColoredMdp, choice_to_assignment: list[list[tuple[int, int]]]) -> tuple[int, int] | None:
    """A (state, action) enabled in some environments only, or None.

    Assumes no _shared_action(): the choices of an action then enable pairwise disjoint sets of environments, so counting those decides whether they cover all.
    """
    parameter_space = colored_mdp.parameter_space
    size = parameter_space.size
    num_options = [parameter_space.parameter_num_options(parameter) for parameter in range(parameter_space.num_parameters)]
    for state, action_choices in enumerate(colored_mdp.feature_info.state_action_choices):
        for action, choices in enumerate(action_choices):
            enabled_in = sum(size // math.prod(num_options[parameter] for parameter, _ in choice_to_assignment[choice]) for choice in choices)
            if choices and enabled_in != size:
                return state, action
    return None


def add_policy_parameters(colored_mdp: paynt.colored_mdp.ColoredMdp) -> tuple[paynt.colored_mdp.ColoredMdp, list[int]]:
    """:returns: (the policy-augmented ColoredMdp, indices of the original environment parameters)
    :raises ValueError: unless each environment enables exactly one choice of each action of a state"""
    assert colored_mdp.feature_kind == "family", "policy parameters can only be added to an MDP family"
    paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, "robust synthesis over an MDP family (adding policy parameters)")
    info = colored_mdp.feature_info
    mdp = colored_mdp.underlying_mdp
    parameter_space = colored_mdp.parameter_space.copy()
    # copy() shares the name/label lists with the original: unshare them before adding parameters to this copy only
    parameter_space.parameter_to_name = list(parameter_space.parameter_to_name)
    parameter_space.parameter_to_option_labels = list(parameter_space.parameter_to_option_labels)
    environment_parameters = list(range(parameter_space.num_parameters))

    choice_to_assignment = colored_mdp.coloring.getChoiceToAssignment()
    state_names = _state_names(mdp)
    # A policy picks actions, so otherwise the augmented model would violate Definition 2 of arXiv:2511.08078: a full assignment (environment and policy)
    # would not induce a Markov chain. With several choices of the picked action enabled, the model stays nondeterministic; with none, it deadlocks. The
    # family itself only colors the environment's choices (the policy tree leaves the policy open), so this is the first point where it is checked.
    requirement = "robust synthesis needs each environment to enable exactly one choice of each action of a state"
    shared = _shared_action(colored_mdp, choice_to_assignment)
    if shared is not None:
        state, action = shared
        raise ValueError(
            f"{requirement}, but in state [{state_names[state]}] some environment enables several choices of action {info.action_labels[action]!r}"
        )
    partial = _partial_action(colored_mdp, choice_to_assignment)
    if partial is not None:
        state, action = partial
        raise ValueError(f"{requirement}, but in state [{state_names[state]}] action {info.action_labels[action]!r} is enabled in some environments only")

    nci = mdp.nondeterministic_choice_indices
    for state in range(mdp.nr_states):
        actions = info.state_to_actions[state]
        if len(actions) < 2:
            continue
        parameter = parameter_space.num_parameters
        parameter_space.add_parameter(f"A([{state_names[state]}])", [info.action_labels[action] for action in actions])
        for choice in range(nci[state], nci[state + 1]):
            choice_to_assignment[choice].append((parameter, actions.index(info.choice_to_action[choice])))

    coloring = payntbind.synthesis.Coloring(parameter_space.native, nci, choice_to_assignment)
    # "generic": with the policy fixed as well, a full assignment induces a DTMC
    augmented = paynt.colored_mdp.ColoredMdp(mdp, parameter_space, coloring, colored_mdp.use_exact, feature_kind="generic")
    return augmented, environment_parameters
