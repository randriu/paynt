"""Conditional reachability properties P[F target || F condition]: the probability of reaching target given that condition is reached.

The value is undefined where condition is unreachable. A design that never reaches it violates the property, and never improves an objective.

Storm checks such a property for the initial state only. Its best scheduler for an MDP has memory. A path is in phase 0 until it enters condition (phase 1, then
it still needs target) or target (phase 2, then it still needs condition), see payntbind's Conditional.h. A design of a colored MDP uses the same parameter
assignment in every phase. Under --conditional-parameters per-phase, phases 1 and 2 get their own copies of the parameters instead (see unfold_phases): a policy
that observes which set it entered may then change its choices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import stormpy
import payntbind

import paynt.specification.property
import paynt.specification.property_result

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import paynt.colored_mdp
    import paynt.model.model

#: --conditional-parameters: one parameter assignment for every phase, or a copy of the parameters for each phase after the first
CONDITIONAL_PARAMETERS = ["shared", "per-phase"]

#: the name suffix of the copy of a parameter for phase 1 and phase 2
PHASE_SUFFIXES = {1: "@condition", 2: "@target"}


def target_and_condition(model: Any, prop: paynt.specification.property.Property) -> tuple[Any, Any]:
    """The states of the model that satisfy the target and the condition of prop (bit vectors)."""
    conditional = prop.formula.subformula
    target = stormpy.model_checking(model, conditional.main_subformula.subformula, force_fully_observable=True).get_truth_values_vector()
    condition = stormpy.model_checking(model, conditional.conditional_subformula.subformula, force_fully_observable=True).get_truth_values_vector()
    return target, condition


def model_check(
    mdp: paynt.model.model.Mdp, prop: paynt.specification.property.Property, alt: bool = False
) -> paynt.specification.property_result.PropertyResult:
    """Check prop on mdp: in its direction (the best case), or with alt in the opposite one (the worst case).

    The value is undefined if the condition is not reached: in the best case, if no scheduler reaches it; in the worst case, if one avoids it, as it then
    violates prop. A constraint is checked on an MDP with its bound, so that Storm decides the threshold without computing the value, which is much faster.
    A Markov chain is checked as an MDP: Storm's procedure for Markov chains crashes on some. A bound outside (0, 1), which Storm's threshold decision may never
    finish deciding, is compared with the value.
    """
    _target, condition = target_and_condition(mdp.model, prop)
    if alt:
        undefined = payntbind.synthesis.some_scheduler_avoids(mdp.model, condition)
    else:
        undefined = not payntbind.synthesis.some_scheduler_reaches(mdp.model, condition)
    if undefined:
        return paynt.specification.property_result.PropertyResult(prop, None, None, sat=False)

    if isinstance(prop, paynt.specification.property.OptimalityProperty) or mdp.is_deterministic or not 0 < prop.threshold < 1:
        formula = prop.formula_alt if alt else prop.formula
    else:
        formula = prop.formula_with_bound_alt if alt else prop.formula_with_bound
    model = payntbind.synthesis.dtmc_as_mdp(mdp.model) if stormpy.models.SparseDtmc.is_instantiation(mdp.model) else mdp.model
    result = paynt.specification.property.Property.model_check(model, formula, only_initial_states=True)
    if result.is_quantitative():
        # Storm computes the value with the bound as well when it is easy to find
        return paynt.specification.property_result.PropertyResult(prop, result, result.get_value(mdp.initial_state))
    return paynt.specification.property_result.PropertyResult(prop, result, None, sat=result.get_truth_values_vector().get(mdp.initial_state))


def scheduler_choices(mdp: paynt.model.model.SubMdp, prop: paynt.specification.property.Property, scheduler: Any) -> dict[int, int]:
    """The choices of the underlying MDP that a scheduler of the sub-MDP for prop takes where they matter (in every phase), each with the length of a shortest
    path to it."""
    target, condition = target_and_condition(mdp.model, prop)
    return payntbind.synthesis.conditional_scheduler_choices(mdp.model, scheduler, target, condition, mdp.underlying_mdp_choice_map)


def scheduler_choice_mask(underlying_mdp: Any, mdp: paynt.model.model.SubMdp, prop: paynt.specification.property.Property, scheduler: Any) -> Any:
    """scheduler_choices as a bit vector over the choices of the underlying MDP."""
    return stormpy.datastructures.BitVector(underlying_mdp.nr_choices, list(scheduler_choices(mdp, prop, scheduler)))


def attains_best_case(prop: paynt.specification.property.Property, best_case: Any, value: Any) -> bool:
    """Whether the value of a design is the best case it comes from, up to the model checking precision.

    Storm's scheduler for a conditional property may not attain its value (see payntbind's Conditional.h), and then neither does the design of a consistent one.
    """
    if value is None:
        return False
    if prop.use_exact:
        return bool(value == best_case)
    return abs(value - best_case) <= paynt.specification.property.Property.model_checking_precision


def backward_scores(
    colored_mdp: paynt.colored_mdp.ColoredMdp,
    mdp: paynt.model.model.SubMdp,
    prop: paynt.specification.property.Property,
    scheduler: Any,
    inconsistent_parameters: dict[int, list[int]],
) -> dict[int, float]:
    """Score each inconsistent parameter by the farthest choice of the scheduler that uses it (the length of a shortest path to it), so that the farthest
    is split first: the backward refinement of Section 5 of arXiv:2605.11897, as Storm gives no state values for a conditional property."""
    colors = colored_mdp.choice_colors()
    scores: dict[int, float] = {}
    for choice, depth in scheduler_choices(mdp, prop, scheduler).items():
        for parameter, _option in colors[choice]:
            if parameter in inconsistent_parameters and depth >= scores.get(parameter, -1):
                scores[parameter] = depth
    return scores


def unfold_phases(colored_mdp: paynt.colored_mdp.ColoredMdp, specification: paynt.specification.property.Specification) -> paynt.colored_mdp.ColoredMdp:
    """The colored MDP whose designs choose the parameters anew in phases 1 and 2 of the conditional property of the specification (--conditional-parameters
    per-phase).

    Its MDP is the product of the MDP with the phases. A choice in phase 1 or 2 is colored by copies of its parameters, named with the suffixes in
    PHASE_SUFFIXES, with the options the parameter has.
    """
    # local: paynt.colored_mdp imports this module
    import paynt.colored_mdp
    import paynt.utils.error_handling

    prop = paynt.utils.error_handling.require_single_conditional(specification, "--conditional-parameters per-phase")
    paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, "--conditional-parameters per-phase")
    target, condition = target_and_condition(colored_mdp.underlying_mdp, prop)
    product, choice_map, choice_phase = payntbind.synthesis.unfold_conditional_phases(colored_mdp.underlying_mdp, target, condition)
    colors = colored_mdp.choice_colors()

    parameter_space = colored_mdp.parameter_space.copy()
    # copy() shares the name/label lists with the original: unshare them before adding parameters to this copy only
    parameter_space.parameter_to_name = list(parameter_space.parameter_to_name)
    parameter_space.parameter_to_option_labels = list(parameter_space.parameter_to_option_labels)
    copied = sorted({(phase, parameter) for choice, phase in zip(choice_map, choice_phase, strict=True) if phase != 0 for parameter, _ in colors[choice]})
    copy_of: dict[tuple[int, int], int] = {}
    for phase, parameter in copied:
        copy_of[phase, parameter] = parameter_space.num_parameters
        parameter_space.add_parameter(parameter_space.parameter_name(parameter) + PHASE_SUFFIXES[phase], parameter_space.parameter_to_option_labels[parameter])
        parameter_space.parameter_set_options(copy_of[phase, parameter], parameter_space.parameter_options(parameter))

    choice_to_assignment = [
        [(copy_of[phase, parameter] if phase != 0 else parameter, option) for parameter, option in colors[choice]]
        for choice, phase in zip(choice_map, choice_phase, strict=True)
    ]
    coloring = payntbind.synthesis.Coloring(parameter_space.native, product.nondeterministic_choice_indices, choice_to_assignment)
    logger.info(
        f"unfolded the phases of {str(prop).strip()}: {product.nr_states} states and {product.nr_choices} choices, {len(copied)} parameters copied for the "
        "phases after the condition or the target is reached"
    )
    return paynt.colored_mdp.ColoredMdp(product, parameter_space, coloring, colored_mdp.use_exact)
