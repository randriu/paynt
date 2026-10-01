"""Shared Z3 BitVec construction for parameter spaces.

Used both by SynthesizerSMPMC's theory-solver encoding (paynt/synthesizer/smpmc/encoding.py) and by CEGIS's SmtSolver (paynt/parameter_space/smt.py), so the two
engines' custom-constraint framework (paynt/parameter_space/constraints/) only has to deal with one variable sort.

BitVec, not Int: z3.UserPropagateBase's `fixed` callback does not fire for Int-sorted variables registered via `.add()`, only for Boolean literals -- SMPMC's
theory-solver integration requires BitVec. CEGIS does not use UserPropagateBase and would work equally well with Int, but shares this module anyway so both
engines speak the same sort to the constraint framework.
"""

from __future__ import annotations

import math
from typing import Any

import z3

import paynt.parameter_space.parameter_space


def bitvec_width(parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> int:
    """Smallest bit width, shared across every parameter, that can represent any parameter's full option count, plus one spare bit of headroom above that."""
    if parameter_space.num_parameters == 0:
        return 1
    max_options = max(parameter_space.parameter_num_options_total(parameter) for parameter in range(parameter_space.num_parameters))
    return math.ceil(math.log2(max_options + 1)) + 1


def parameter_bitvec_variables(
    parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
) -> tuple[list[Any], int, dict[str, int]]:
    """One z3.BitVec per parameter, of a shared width sized for the whole parameter_space.

    Each variable is named f"{parameter_name}#{index}", not the bare parameter name: parameter names are not guaranteed unique (POMDP/DT factories generate
    names from observation expressions), and Z3 interns BitVecs by name+sort, so two same-named parameters would otherwise silently become the same Z3 variable.

    :returns: (variables, width, name_to_parameter), where name_to_parameter maps each variable's str() back to its parameter index.
    """
    width = bitvec_width(parameter_space)
    variables: list[Any] = []
    name_to_parameter: dict[str, int] = {}
    for parameter in range(parameter_space.num_parameters):
        var_name = f"{parameter_space.parameter_name(parameter)}#{parameter}"
        var = z3.BitVec(var_name, width)
        variables.append(var)
        name_to_parameter[var_name] = parameter
    return variables, width, name_to_parameter
