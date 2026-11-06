"""Tests of paynt.parameter_space.smt (SmtSolver/ParameterSpaceEncoding): the BitVec encoding of a parameter space (see paynt/parameter_space/bitvec.py) and the
shared custom-constraint hook (paynt/parameter_space/constraints/)."""

from __future__ import annotations

import z3

import paynt.parameter_space.constraints
import paynt.parameter_space.parameter_space
import paynt.parameter_space.smt
import paynt.synthesizer.search_node


def _parameter_space(*option_counts):
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for i, count in enumerate(option_counts):
        parameter_space.add_parameter(f"p{i}", [str(o) for o in range(count)])
    return parameter_space


class TestSmtSolverVariables:
    def test_variables_are_bitvec(self):
        solver = paynt.parameter_space.smt.SmtSolver(_parameter_space(3, 2))
        assert all(isinstance(v, z3.BitVecRef) for v in solver.solver_vars)

    def test_one_clause_per_parameter_option(self):
        parameter_space = _parameter_space(3, 2)
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        assert len(solver.solver_clauses) == 2
        assert len(solver.solver_clauses[0]) == 3
        assert len(solver.solver_clauses[1]) == 2

    def test_no_constraint_means_no_constraint_clauses(self):
        solver = paynt.parameter_space.smt.SmtSolver(_parameter_space(2))
        assert solver.constraint_clauses == []


class TestParameterSpaceEncodingPickAssignment:
    def test_picks_a_valid_assignment(self):
        parameter_space = _parameter_space(3, 2)
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, parameter_space)
        assignment = encoding.pick_assignment()
        assert assignment is not None
        for parameter in range(parameter_space.num_parameters):
            assert assignment.parameter_options(parameter)[0] in parameter_space.parameter_options(parameter)

    def test_respects_a_narrowed_parameter_space(self):
        parameter_space = _parameter_space(3)
        narrowed = parameter_space.assume_parameter_options_copy(0, [1])
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, narrowed)
        assignment = encoding.pick_assignment()
        assert assignment.parameter_options(0) == [1]

    def test_exhausts_to_none(self):
        parameter_space = _parameter_space(1)  # a single parameter, single option: exactly one assignment
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, parameter_space)
        first = encoding.pick_assignment()
        assert first is not None
        solver.solver.add(z3.Not(solver.create_parameter_clause(0, 0)))
        assert encoding.pick_assignment() is None


class TestSmtSolverConstraintHook:
    def test_constraint_clauses_are_conjoined_into_the_encoding(self):
        parameter_space = _parameter_space(4)

        class _OnlyOptionTwo(paynt.parameter_space.constraints.Constraint):
            name = "test-only-two"

            def build(self, ctx):
                return [ctx.eq(0, 2)]

        # the constraint reads neither the colored MDP nor the task
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space, _OnlyOptionTwo(), colored_mdp="unused", task="unused")
        assert len(solver.constraint_clauses) == 1
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, parameter_space)
        assignment = encoding.pick_assignment()
        assert assignment.parameter_options(0) == [2]
        solver.solver.add(z3.Not(solver.create_parameter_clause(0, 2)))
        assert encoding.pick_assignment() is None


def _remaining(solver, parameter_space):
    """The full assignments the solver still allows."""
    remaining = set()
    for combination in parameter_space.all_combinations():
        clauses = [solver.solver_clauses[parameter][option] for parameter, option in enumerate(combination)]
        if solver.solver.check(z3.And(clauses)) == z3.sat:
            remaining.add(tuple(combination))
    return remaining


class TestSmtSolverExcludeConflict:
    """A space of p0 in {0,1} and p1 in {0,1,2}, and the conflict of the assignment p0=0, p1=0."""

    def _exclude(self, node_space, conflict):
        parameter_space = _parameter_space(2, 3)
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        node = paynt.synthesizer.search_node.SearchNode(node_space(parameter_space))
        node.encode(solver)
        assignment = parameter_space.assume_options_copy([[0], [0]])
        pruning_estimate = solver.exclude_conflict(node, assignment, conflict)
        return pruning_estimate, _remaining(solver, parameter_space)

    def test_a_conflict_of_every_parameter_removes_only_the_assignment(self):
        pruning_estimate, remaining = self._exclude(lambda space: space, conflict=[0, 1])
        assert pruning_estimate == 1
        assert remaining == {(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}

    def test_a_smaller_conflict_removes_every_assignment_that_agrees_on_it(self):
        pruning_estimate, remaining = self._exclude(lambda space: space, conflict=[0])
        assert pruning_estimate == 3
        assert remaining == {(1, 0), (1, 1), (1, 2)}

    def test_the_removal_stays_inside_the_node(self):
        pruning_estimate, remaining = self._exclude(lambda space: space.assume_parameter_options_copy(1, [0, 1]), conflict=[0])
        assert pruning_estimate == 2
        assert remaining == {(0, 2), (1, 0), (1, 1), (1, 2)}
