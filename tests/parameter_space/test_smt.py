"""Backfill for paynt.parameter_space.smt (SmtSolver/ParameterSpaceEncoding), previously untested.

Written while migrating SmtSolver from z3.Int to z3.BitVec (see paynt/parameter_space/bitvec.py) and adding the shared custom-constraint hook
(paynt/parameter_space/constraints/).
"""

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
        # exclude every option except 2 via a constraint, and confirm pick_assignment only ever returns 2
        var_wrapper = {}

        class _OnlyOptionTwo(paynt.parameter_space.constraints.Constraint):
            name = "test-only-two"

            def build(self, ctx):
                var_wrapper["var"] = ctx.variables[0]
                return [ctx.eq(0, 2)]

        # this constraint never touches colored_mdp/task, so any non-None placeholder satisfies
        # SmtSolver's "a constraint needs both" assertion without needing a real ColoredMdp/SynthesisTask
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space, _OnlyOptionTwo(), colored_mdp="unused", task="unused")
        assert len(solver.constraint_clauses) == 1
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, parameter_space)
        assignment = encoding.pick_assignment()
        assert assignment.parameter_options(0) == [2]
        # and no other option is ever reachable, no matter how many times we ask
        for _ in range(3):
            assignment = encoding.pick_assignment()
            if assignment is None:
                break
            assert assignment.parameter_options(0) == [2]

    def test_no_constraint_behaves_exactly_as_before(self):
        """Regression guard: SmtSolver's pre-Phase-2 call shape (positional parameter_space only) must keep working, since CEGIS/Hybrid predate the constraint
        parameter."""
        parameter_space = _parameter_space(3)
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, parameter_space)
        assert encoding.pick_assignment() is not None


class TestSmtSolverExcludeConflict:
    def test_excluding_a_conflict_removes_that_assignment(self):
        parameter_space = _parameter_space(2)
        solver = paynt.parameter_space.smt.SmtSolver(parameter_space)
        node = paynt.synthesizer.search_node.SearchNode(parameter_space)
        node.encode(solver)  # exclude_conflict assumes the node is already encoded, as it is mid-CEGIS-loop
        assignment = parameter_space.assume_parameter_options_copy(0, [0])
        pruning_estimate = solver.exclude_conflict(node, assignment, conflict=[0])
        assert pruning_estimate == 1
        # the excluded option should no longer be reachable
        encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, parameter_space)
        for _ in range(2):
            found = encoding.pick_assignment()
            if found is None:
                break
            assert found.parameter_options(0) != [0]
