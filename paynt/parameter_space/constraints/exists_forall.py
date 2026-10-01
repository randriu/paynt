"""ExistsForallConstraint: robust synthesis (arXiv:2511.08078, Section 4.4) -- exists a choice of the non-quantified parameters such that viable(...) holds for
every in-range choice of ctx.forall_parameters. Z3 decides this via MBQI; the theory solver needs nothing quantifier-specific.

Ported from molehill's constraints/exists_forall.py (https://github.com/linusheck/molehill, GPL-3.0).
"""

from __future__ import annotations

from typing import Any

import z3

# direct name import, see exists.py
from paynt.parameter_space.constraints.constraint import Constraint, ConstraintContext


class ExistsForallConstraint(Constraint):
    name = "exists_forall"

    def build(self, ctx: ConstraintContext) -> list[Any]:
        viable = ctx.requires_viable()
        if not ctx.forall_parameters:
            raise ValueError("exists_forall needs at least one universally quantified parameter")
        forall_variables = [ctx.variables[parameter] for parameter in ctx.forall_parameters]
        in_range = ctx.in_range()
        clauses = [in_range, z3.ForAll(forall_variables, z3.Implies(in_range, viable(*ctx.variables)))]
        # MBQI only instantiates quantifiers with values occurring in ground terms of the formula, so put every
        # value the variables can take in scope (molehill does the same)
        for value in range(2**ctx.width):
            clauses.append(z3.BitVec(f"__forall_value_{value}", ctx.width) == ctx.value(value))
        return clauses
