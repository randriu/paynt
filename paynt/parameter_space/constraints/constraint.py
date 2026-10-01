"""Constraint: the shared extension point for custom SMT constraints over the parameter space, usable by both SynthesizerSMPMC (natively -- a constraint can
reference the viable(...) predicate, e.g. to wrap it in a quantifier for robust synthesis) and CEGIS's SmtSolver (structural constraints only, no viable -- see
ConstraintContext.requires_viable).

Ported from molehill's constraints/constraint.py (https://github.com/linusheck/molehill, GPL-3.0), as a
redesign rather than a direct copy: molehill couples this to argparse (register_arguments/set_args), but
PAYNT threads per-run configuration through SynthesisTask instead, matching every other engine knob (e.g.
SynthesisTask.conflict_generator_type).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import z3

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.task

import logging

logger = logging.getLogger(__name__)


@dataclass
class ConstraintContext:
    """Everything a Constraint needs to emit Z3 clauses.

    One shape for both engines: CEGIS's SmtSolver and SMPMC's ParameterBitVecEncoding both build one z3.BitVec per parameter through the same
    paynt.parameter_space.bitvec helper, so there is no per-engine variable-sort split to abstract over.
    """

    colored_mdp: paynt.colored_mdp.ColoredMdp
    parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
    task: paynt.task.SynthesisTask
    variables: list[Any]
    width: int
    # the z3.PropagateFunction registered by SMPMC's theory-solver plugin; None when called from CEGIS,
    # which has no theory-solver integration for a constraint to hook a refutation into
    viable: Any | None = None
    # parameters universally quantified by a robust (exists-forall) constraint; empty for plain existential search
    forall_parameters: list[int] = field(default_factory=list)

    def value(self, option: int) -> Any:
        return z3.BitVecVal(option, self.width)

    def eq(self, parameter: int, option: int) -> Any:
        return self.variables[parameter] == self.value(option)

    def in_range(self) -> Any:
        """tau_V's domain conjunct: every parameter constrained to one of its currently-allowed options."""
        clauses = []
        for parameter in range(self.parameter_space.num_parameters):
            option_clauses = [self.eq(parameter, option) for option in self.parameter_space.parameter_options(parameter)]
            clauses.append(option_clauses[0] if len(option_clauses) == 1 else z3.Or(option_clauses))
        return clauses[0] if len(clauses) == 1 else z3.And(clauses)

    def base_clauses(self) -> list[Any]:
        """in_range(), plus viable(...) when available -- every concrete Constraint is self-contained (mirroring molehill's own constraints, which each
        independently include their equivalent of these two rather than composing with a separate "exists" constraint), so every Constraint.build() should start
        its returned list from this rather than reimplementing it."""
        clauses: list[Any] = [self.in_range()]
        if self.viable is not None:
            clauses.append(self.viable(*self.variables))
        return clauses

    def requires_viable(self) -> Any:
        """Call from a constraint that needs to reference the viable(...) predicate (e.g. to wrap it in a quantifier for robust synthesis).

        Raises a clear, actionable error rather than a confusing AttributeError/TypeError when used from an engine (CEGIS) that has no such predicate.
        """
        if self.viable is None:
            raise ValueError(
                "this constraint requires the SMPMC engine (--method smpmc): it references the "
                "viable(...) predicate, which only SMPMC's theory-solver integration provides"
            )
        return self.viable


class Constraint:
    """Base class for a custom constraint over the parameter space, contributing additional Z3 clauses beyond the plain per-parameter domain
    (ConstraintContext.in_range()).

    Register a new constraint in paynt/parameter_space/constraints/_registry.py's CONSTRAINTS mapping.
    """

    name: ClassVar[str]

    def build(self, ctx: ConstraintContext) -> list[Any]:
        """Return the additional Z3 clauses this constraint contributes.

        Always a list (never a bare expression, unlike molehill's inconsistent build_constraint), consumed via solver.add(*clauses) (SMPMC) or z3.And(encoding,
        *clauses) (CEGIS).
        """
        raise NotImplementedError
