"""Internal support for the "pomdp_family" feature: the PomdpFamilyInfo companion dataclass (attached as ColoredMdp.feature_info for feature_kind
"pomdp_family") and functions that interpret it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import payntbind

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
from paynt.mdp_family._utils import MdpFamilyInfo
import paynt.pomdp.fsc
import paynt.model.model
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)


class SubPomdp:
    """Simple container for a (sub-)POMDP created from the underlying model."""

    def __init__(self, model: Any, underlying_mdp_state_map: list[int], underlying_mdp_choice_map: list[int]):
        # the Stormpy POMDP
        self.model = model
        # for each state of the POMDP, a state in the underlying model
        self.underlying_mdp_state_map = underlying_mdp_state_map
        # for each choice of the POMDP, a choice in the underlying model
        self.underlying_mdp_choice_map = underlying_mdp_choice_map


@dataclass(kw_only=True)
class PomdpFamilyInfo(MdpFamilyInfo):
    obs_evaluator: Any
    # for each observation, a list of actions (indices) available
    observation_to_actions: list[list[int]]
    # set by build_dtmc_sketch; used only within that method
    fsc_unfolder: Any = None


def num_observations(info: PomdpFamilyInfo) -> int:
    return info.obs_evaluator.num_obs_classes


def state_to_observation(info: PomdpFamilyInfo) -> list[int]:
    return info.obs_evaluator.state_to_obs_class


def build_pomdp(colored_mdp: paynt.colored_mdp.ColoredMdp, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> SubPomdp:
    """Construct the sub-POMDP from the given parameter assignment."""
    info = cast(PomdpFamilyInfo, colored_mdp.feature_info)
    assert parameter_space.size == 1, "expecting parameter space of size 1"
    choices = colored_mdp.coloring.selectCompatibleChoices(parameter_space.native)
    mdp, state_map, choice_map = paynt.model.model.SubmodelBuilder.restrict(colored_mdp.underlying_mdp, choices, colored_mdp.subsystem_builder_options)
    pomdp = info.obs_evaluator.add_observations_to_submdp(mdp, state_map)
    return SubPomdp(pomdp, state_map, choice_map)


def build_dtmc_sketch(colored_mdp: paynt.colored_mdp.ColoredMdp, fsc: paynt.pomdp.fsc.Fsc | paynt.pomdp.fsc.FscFactored) -> paynt.colored_mdp.ColoredMdp:
    """Construct the family of DTMCs representing the execution of the given FSC in different environments."""
    paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, "building the FSC product (build_dtmc_sketch)")
    info = cast(PomdpFamilyInfo, colored_mdp.feature_info)

    # create the product
    fsc.check(info.observation_to_actions)

    info.fsc_unfolder = payntbind.synthesis.FscUnfolder(colored_mdp.underlying_mdp, state_to_observation(info), info.num_actions, info.choice_to_action)
    if isinstance(fsc, paynt.pomdp.fsc.Fsc):
        info.fsc_unfolder.applyFsc(fsc.transitions)
    elif isinstance(fsc, paynt.pomdp.fsc.FscFactored):
        info.fsc_unfolder.applyFscFactored(fsc.action_function, fsc.update_function)
    else:
        raise ValueError("unknown FSC class")
    product = info.fsc_unfolder.product
    product_choice_to_choice = info.fsc_unfolder.product_choice_to_choice

    # the product inherits the design space
    product_parameter_space = colored_mdp.parameter_space.copy()

    # the choices of the product inherit colors of the underlying model
    product_choice_to_parameter_options = []
    underlying_num_choices = colored_mdp.underlying_mdp.nr_choices
    choice_to_parameter_assignment = colored_mdp.coloring.getChoiceToAssignment()
    for product_choice in range(product.nr_choices):
        choice = product_choice_to_choice[product_choice]
        if choice == underlying_num_choices:
            parameter_options = []
        else:
            parameter_options = [(parameter, option) for parameter, option in choice_to_parameter_assignment[choice]]
        product_choice_to_parameter_options.append(parameter_options)
    product_coloring = payntbind.synthesis.Coloring(
        product_parameter_space.native, product.nondeterministic_choice_indices, product_choice_to_parameter_options
    )

    return paynt.colored_mdp.ColoredMdp(product, product_parameter_space, product_coloring, use_exact=colored_mdp.use_exact)
