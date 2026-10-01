"""The constraint registry: how the name of --constraint becomes a Constraint."""

from __future__ import annotations

import pytest

import paynt.parameter_space.constraints
from paynt.parameter_space.constraints._registry import CONSTRAINTS


class TestBuildConstraint:
    def test_every_registered_name_builds_a_constraint(self):
        assert sorted(CONSTRAINTS) == ["costs", "exists", "exists_forall", "prob0", "prob1"]
        for name in CONSTRAINTS:
            assert isinstance(paynt.parameter_space.constraints.build_constraint(name), paynt.parameter_space.constraints.Constraint)

    def test_every_call_builds_a_new_constraint(self):
        """A constraint may keep state of a run, so two runs must not share one."""
        first = paynt.parameter_space.constraints.build_constraint("exists")
        assert paynt.parameter_space.constraints.build_constraint("exists") is not first

    def test_the_two_probability_goals_differ_in_the_probability(self):
        zero, one = (paynt.parameter_space.constraints.build_constraint(name) for name in ("prob0", "prob1"))
        assert type(zero) is type(one)
        assert (zero.prob, one.prob) == (0, 1)

    def test_an_unknown_name_lists_the_available_ones(self):
        with pytest.raises(ValueError, match=r"unknown constraint 'bogus'; available: \['costs', 'exists', 'exists_forall', 'prob0', 'prob1'\]"):
            paynt.parameter_space.constraints.build_constraint("bogus")


class TestConstraintBase:
    def test_a_constraint_has_to_say_which_clauses_it_adds(self):
        with pytest.raises(NotImplementedError):
            paynt.parameter_space.constraints.Constraint().build(None)
