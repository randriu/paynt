"""Internal support for the DT feature: the DtInfo companion dataclass (attached as ColoredMdp.feature_info for feature_kind "dt") and functions that interpret
it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import json

import paynt.colored_mdp
import paynt.dt.decision_tree
import paynt.model.model
import paynt.task
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)

# label for action executing a random action selection
DONT_CARE_ACTION_LABEL = "__random__"

# inner search engines DtSynthesizer can run against each tree unfolding, see make_inner_synthesizer
DT_INNER_METHODS = ("ar", "smpmc")


@dataclass(kw_only=True)
class DtInfo:
    # MDP identity, stable across every reset_tree() call (not just this one tree/depth)
    action_labels: list[str]
    choice_to_action: list[int]
    state_is_relevant: list[bool]
    state_is_relevant_bv: Any
    variables: list[Any]
    relevant_state_valuations: list[Any]
    # specific to this tree/depth
    decision_tree: paynt.dt.decision_tree.DecisionTree
    is_action_parameter: list[bool]
    is_decision_parameter: list[bool]
    is_variable_parameter: list[bool]
    # dtnest-only (paynt.dt.dtnest.synthesizer.DtNest): the externally-learned/incrementally-rebuilt tree
    # dtnest works from, distinct from decision_tree above (the AR-search's own tree template). Not part
    # of plain DtSynthesizer's contract -- declared here purely so their type is known at every dtnest
    # read site, not because every "dt" colored MDP genuinely has one.
    tree_helper: Any = None
    tree_helper_tree: paynt.dt.decision_tree.DecisionTree | None = None


# TODO make this so that it works for POMDP observation valuations as well
def get_state_valuations(model: Any) -> tuple[list[str], list[list[Any]]]:
    """Identify variable names and extract state valuation in the same order."""
    assert model.has_state_valuations(), "model has no state valuations"
    # get name
    sv = model.state_valuations
    variable_names: list[str] | None = None
    state_valuations = []
    for state in range(model.nr_states):
        valuation = json.loads(str(sv.get_json(state)))
        if variable_names is None:
            variable_names = list(valuation.keys())
        state_valuations.append([valuation[var_name] for var_name in variable_names])

    assert variable_names is not None
    return variable_names, state_valuations


def simplify_tree(tree: paynt.dt.decision_tree.DecisionTree | None, info: DtInfo) -> None:
    """Simplify the tree recursively by removing irrelavant leaf nodes."""
    if tree is None:
        return

    relevant_state_valuations = [info.relevant_state_valuations[state] for state in info.state_is_relevant_bv]
    tree.simplify(relevant_state_valuations)

    return


def scheduler_json_to_choices(
    colored_mdp: paynt.colored_mdp.ColoredMdp, scheduler_json: list[Any], discard_unreachable_states: bool = False
) -> tuple[Any, list[Any]]:
    info = cast(DtInfo, colored_mdp.feature_info)
    underlying_mdp = colored_mdp.underlying_mdp
    variable_name, state_valuations = get_state_valuations(underlying_mdp)
    nci = underlying_mdp.nondeterministic_choice_indices.copy()
    assert underlying_mdp.nr_states == len(scheduler_json)
    state_to_choice = paynt.model.model.ModelIndex.empty_scheduler(underlying_mdp)
    for state_decision in scheduler_json:
        valuation = [state_decision["s"][name] for name in variable_name]
        for state, state_valuation in enumerate(state_valuations):  # noqa: B007 -- state used below
            if valuation == state_valuation:
                break
        else:
            raise AssertionError("state valuation not found")

        actions = state_decision["c"]
        assert len(actions) == 1
        action_labels = actions[0]["labels"]
        assert len(action_labels) <= 1
        if len(action_labels) == 0:
            state_to_choice[state] = nci[state]
            continue
        action = info.action_labels.index(action_labels[0])
        # find a choice that executes this action
        for choice in range(nci[state], nci[state + 1]):
            if info.choice_to_action[choice] == action:
                state_to_choice[state] = choice
                break
        else:
            raise AssertionError("action is not available in the state")
    # enable implicit actions
    for state, existing_choice in enumerate(state_to_choice):
        if existing_choice is None:
            logger.warning(f"WARNING: scheduler has no action for state {state}")
            state_to_choice[state] = nci[state]

    if discard_unreachable_states:
        state_to_choice = paynt.model.model.ModelIndex.discard_unreachable_choices(underlying_mdp, colored_mdp.choice_destinations, state_to_choice)
    # keep only relevant states
    state_to_choice = [choice if info.state_is_relevant[state] else None for state, choice in enumerate(state_to_choice)]
    choices = paynt.model.model.ModelIndex.state_to_choice_to_choices(underlying_mdp, state_to_choice)

    scheduler_json_relevant = []
    for state_decision in scheduler_json:
        valuation = [state_decision["s"][name] for name in variable_name]
        for state, state_valuation in enumerate(state_valuations):  # noqa: B007 -- state used below
            if valuation == state_valuation:
                break
        if state_to_choice[state] is None:
            continue
        scheduler_json_relevant.append(state_decision)

    return choices, scheduler_json_relevant


def get_random_choices(colored_mdp: paynt.colored_mdp.ColoredMdp) -> Any:
    """Gets all choices that represent random action, used to compute the value of uniformly random scheduler."""
    info = cast(DtInfo, colored_mdp.feature_info)
    underlying_mdp = colored_mdp.underlying_mdp
    nci = underlying_mdp.nondeterministic_choice_indices.copy()
    state_to_choice = paynt.model.model.ModelIndex.empty_scheduler(underlying_mdp)
    random_action = info.action_labels.index(DONT_CARE_ACTION_LABEL)
    for state in range(underlying_mdp.nr_states):
        for choice in range(nci[state], nci[state + 1]):
            if info.choice_to_action[choice] == random_action:
                state_to_choice[state] = choice
                break
    for state, existing_choice in enumerate(state_to_choice):
        if existing_choice is None:
            state_to_choice[state] = nci[state]

    return paynt.model.model.ModelIndex.state_to_choice_to_choices(underlying_mdp, state_to_choice)


def make_inner_synthesizer(method: str, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask) -> Any:
    """Construct the inner search engine DtSynthesizer runs against one tree unfolding.

    :param method: "ar" (SynthesizerARDt, over ColoringSmt) or "smpmc" (SynthesizerSMPMC, over the tree's ColoringGeneral, see paynt.dt.coloring_general)
    """
    paynt.utils.error_handling.require_supported_method(method, DT_INNER_METHODS, "decision-tree synthesis")
    # hiding imports here to avoid mutual top-level imports (mirroring Synthesizer.for_method)
    if method == "ar":
        from paynt.dt.synthesizer_ar_dt import SynthesizerARDt

        return SynthesizerARDt(colored_mdp, task)
    if method == "smpmc":
        from paynt.synthesizer.smpmc import SynthesizerSMPMC

        return SynthesizerSMPMC(colored_mdp, task)
    raise AssertionError(f"DT_INNER_METHODS has a method that make_inner_synthesizer does not build: {method!r}")
