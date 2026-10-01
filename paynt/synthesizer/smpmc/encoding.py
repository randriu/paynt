"""ParameterBitVecEncoding: the BitVec variables of a parameter space, and the clauses and models that SynthesizerSMPMC.synthesize_one exchanges with Z3 through
them.

The domain (tau_V) clause of the parameters is ConstraintContext.in_range.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import z3

import paynt.parameter_space.bitvec
import paynt.parameter_space.parameter_space

import logging

logger = logging.getLogger(__name__)


class ParameterBitVecEncoding:
    def __init__(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace):
        self.parameter_space = parameter_space
        self.variables, self.width, self.name_to_parameter = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)

    def value(self, option: int) -> Any:
        return z3.BitVecVal(option, self.width)

    def exclude(self, assignment: paynt.parameter_space.parameter_space.ParameterSpace, parameters: Iterable[int] | None = None) -> Any:
        """A clause ruling out assignment's options of parameters (by default all of them, a single concrete assignment) -- used by the optimality loop to force
        Z3 to look elsewhere after recording assignment as the new best, since nothing else about the Z3-level formula changes when only the theory's threshold
        tightens."""
        if parameters is None:
            parameters = range(assignment.num_parameters)
        equalities = []
        for parameter in parameters:
            options = assignment.parameter_options(parameter)
            assert len(options) == 1, "exclude() expects every excluded parameter fixed"
            equalities.append(self.variables[parameter] == self.value(options[0]))
        conjunction = equalities[0] if len(equalities) == 1 else z3.And(equalities)
        return z3.Not(conjunction)

    def extract_assignment(
        self,
        model: Any,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        free_parameters: Iterable[int] = (),
    ) -> paynt.parameter_space.parameter_space.ParameterSpace:
        """:param free_parameters: parameters left at their options in parameter_space rather than fixed to their value in
        model -- the universally quantified parameters of a robust result, whose model values mean nothing"""
        free = set(free_parameters)
        parameter_options = []
        for parameter in range(parameter_space.num_parameters):
            if parameter in free:
                parameter_options.append(parameter_space.parameter_options(parameter))
                continue
            option = model.eval(self.variables[parameter], model_completion=True).as_long()
            parameter_options.append([option])
        return parameter_space.assume_options_copy(parameter_options)
