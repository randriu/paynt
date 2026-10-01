from __future__ import annotations

from typing import Any

import stormpy

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.synthesizer.synthesizer_ar
import paynt.task

import logging

logger = logging.getLogger(__name__)


def verify_robust(
    colored_mdp: paynt.colored_mdp.ColoredMdp,
    task: paynt.task.SynthesisTask,
    robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace,
    value: Any = None,
) -> bool:
    """Check a robust result independently of Z3, as molehill's tests/test_robust.py does: robust_assignment is robust iff AR finds none of its members to
    satisfy the negated specification.

    :param value: for an optimality objective, the claimed worst case over robust_assignment, checked as the threshold "at least as good as value"
    """
    specification = task.specification
    if specification.optimality is not None:
        assert value is not None, "verifying a robust optimum needs its value"
        specification = _threshold_specification(specification.optimality, value)
    negated_task = paynt.task.SynthesisTask.from_specification(specification.negate(), use_exact=task.use_exact)
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(colored_mdp, negated_task)
    counterexample = synthesizer.synthesize(robust_assignment, print_stats=False)
    if counterexample is not None:
        logger.error(f"robustness check failed, this member violates the specification: {counterexample}")
    return counterexample is None


def _threshold_specification(optimality: paynt.specification.property.OptimalityProperty, value: Any) -> paynt.specification.property.Specification:
    """The objective as a threshold: at least as good as value, relaxed by the model checking precision (value itself is only that precise)."""
    formula = optimality.property.raw_formula.clone()
    bound = value
    if not optimality.use_exact:
        tolerance = paynt.specification.property.Property.model_checking_precision * max(1, abs(value))
        bound = value + tolerance if optimality.minimizing else value - tolerance
    comparison = stormpy.ComparisonType.LEQ if optimality.minimizing else stormpy.ComparisonType.GEQ
    formula.set_bound(comparison, stormpy.ExpressionManager().create_rational(stormpy.Rational(bound)))
    formula.remove_optimality_type()
    prop = paynt.specification.property.Property(stormpy.Property("", formula), optimality.use_exact)
    prop.discount_factor_correction = optimality.discount_factor_correction
    return paynt.specification.property.Specification([prop])


def worst_case_by_ar(
    colored_mdp: paynt.colored_mdp.ColoredMdp,
    optimality: paynt.specification.property.OptimalityProperty,
    robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace,
) -> Any:
    """The worst case of the objective over robust_assignment's members, by AR on the opposite objective -- with relative error 0: stopping early could
    overestimate the worst case."""
    opposite = optimality.negate()
    exact_opposite = paynt.specification.property.OptimalityProperty(opposite.property, 0, optimality.use_exact)
    exact_opposite.discount_factor_correction = opposite.discount_factor_correction
    task = paynt.task.SynthesisTask.from_specification(paynt.specification.property.Specification([exact_opposite]), use_exact=optimality.use_exact)
    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(colored_mdp, task)
    synthesizer.synthesize(robust_assignment, keep_optimum=True, print_stats=False)
    return synthesizer.best_assignment_value
