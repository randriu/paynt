"""ProbGoalConstraint (--constraint prob0/prob1) on the sketches of models/tests/prob-goal, whose comments give the reachability probability of every
assignment.

Their property, P>=0, holds everywhere, so the engines are restricted by the constraint alone.
"""

from __future__ import annotations

import pytest
import z3

import paynt.parameter_space.bitvec
import paynt.parameter_space.constraints
import paynt.synthesizer.synthesizer
from paynt.parameter_space.constraints.constraint import ConstraintContext

from tests.helpers.models import load_colored_mdp


def accepted_assignments(colored_mdp, task, constraint_name):
    """The full assignments that the constraint's clauses admit."""
    space = colored_mdp.parameter_space
    variables, width, _ = paynt.parameter_space.bitvec.parameter_bitvec_variables(space)
    ctx = ConstraintContext(colored_mdp=colored_mdp, parameter_space=space, task=task, variables=variables, width=width)
    solver = z3.Solver()
    solver.add(*paynt.parameter_space.constraints.build_constraint(constraint_name).build(ctx))
    return {
        str(space.construct_assignment(options))
        for options in space.all_combinations()
        if solver.check(*[variable == z3.BitVecVal(option, width) for variable, option in zip(variables, options, strict=True)]) == z3.sat
    }


class TestClauses:
    @pytest.mark.parametrize(
        "sketch_name, constraint_name, expected",
        [
            ("sketch.templ", "prob0", {"a=2"}),
            ("sketch.templ", "prob1", set()),
            # the agent's choice: some resolution reaches the goal (prob0), every resolution reaches it almost surely (prob1)
            ("family.templ", "prob0", {"a=0", "a=1"}),
            ("family.templ", "prob1", {"a=0"}),
        ],
    )
    def test_admit_exactly_the_assignments_reaching_the_goal(self, sketch_name, constraint_name, expected):
        colored_mdp, task = load_colored_mdp("tests/prob-goal", sketch_name=sketch_name)
        assert accepted_assignments(colored_mdp, task, constraint_name) == expected

    @pytest.mark.parametrize("constraint_name", ["prob0", "prob1"])
    def test_admit_exactly_the_assignments_reaching_the_goal_on_smpmc_tiny(self, smpmc_tiny, constraint_name):
        # every assignment reaches "done" with probability 0 or 1
        expected = {"x1=1, x2=4, x3=8", "x1=4, x2=3, x3=8", "x1=4, x2=4, x3=8"}
        assert accepted_assignments(smpmc_tiny.colored_mdp, smpmc_tiny.task, constraint_name) == expected


@pytest.mark.parametrize("method", ["smpmc", "cegis"])
class TestEngines:
    @pytest.mark.parametrize(
        "sketch_name, constraint_name, expected",
        [
            ("sketch.templ", "prob0", "a=2"),
            ("sketch.templ", "prob1", None),
            ("family.templ", "prob1", "a=0"),
        ],
    )
    def test_find_the_only_admitted_assignment(self, method, sketch_name, constraint_name, expected):
        colored_mdp, task = load_colored_mdp("tests/prob-goal", sketch_name=sketch_name)
        task.constraint_name = constraint_name
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        if expected is None:
            assert not result.success
        else:
            assert result.success
            assert str(result.assignment) == expected
