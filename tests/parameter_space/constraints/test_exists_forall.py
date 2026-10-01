"""Unit tests of ExistsForallConstraint's formula -- no Storm.

Its semantics under MBQI are covered by tests/synthesizer/smpmc/test_propagator.py (fake theory) and tests/synthesizer/smpmc/test_robust.py (real models).
"""

from __future__ import annotations

import pytest
import z3

import paynt.parameter_space.bitvec
import paynt.parameter_space.constraints
import paynt.parameter_space.parameter_space


def _context(forall_parameters, with_viable=True):
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    parameter_space.add_parameter("x", [0, 1, 2])
    parameter_space.add_parameter("y", [0, 1])
    parameter_space.add_parameter("z", [0, 1, 2, 3])
    variables, width, _ = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)
    viable = z3.Function("viable", *[variable.sort() for variable in variables], z3.BoolSort()) if with_viable else None
    return paynt.parameter_space.constraints.ConstraintContext(
        colored_mdp=None,  # type: ignore[arg-type]
        parameter_space=parameter_space,
        task=None,  # type: ignore[arg-type]
        variables=variables,
        width=width,
        viable=viable,
        forall_parameters=forall_parameters,
    )


def _quantifier(clauses):
    quantifiers = [clause for clause in clauses if z3.is_quantifier(clause)]
    assert len(quantifiers) == 1
    return quantifiers[0]


class TestExistsForallConstraint:
    def test_quantifies_exactly_the_forall_parameters(self):
        ctx = _context(forall_parameters=[1, 2])
        quantifier = _quantifier(paynt.parameter_space.constraints.build_constraint("exists_forall").build(ctx))
        assert quantifier.is_forall()
        bound = [quantifier.var_name(i) for i in range(quantifier.num_vars())]
        assert bound == [str(ctx.variables[1]), str(ctx.variables[2])]

    def test_every_bitvector_value_is_put_in_scope_for_mbqi(self):
        ctx = _context(forall_parameters=[0])
        clauses = paynt.parameter_space.constraints.build_constraint("exists_forall").build(ctx)
        ground_values = {arg.as_long() for clause in clauses if z3.is_eq(clause) for arg in clause.children() if z3.is_bv_value(arg)}
        assert ground_values == set(range(2**ctx.width))

    def test_requires_a_forall_parameter(self):
        with pytest.raises(ValueError, match="universally quantified"):
            paynt.parameter_space.constraints.build_constraint("exists_forall").build(_context(forall_parameters=[]))

    def test_rejected_without_the_smpmc_theory(self):
        """CEGIS has no viable(...) predicate to quantify over: this must fail loudly, not degrade to a plain search."""
        with pytest.raises(ValueError, match="--method smpmc"):
            paynt.parameter_space.constraints.build_constraint("exists_forall").build(_context(forall_parameters=[0], with_viable=False))
