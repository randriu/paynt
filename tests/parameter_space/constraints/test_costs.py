"""Integration tests for CostsConstraint against models/tests/smpmc-tiny/sketch.costs, whose costs were
chosen by hand so the threshold boundary is exact and unambiguous:

    x1 costs: [10, 20, 30, 40]  (options 0-3, labels 1-4)
    x2 costs: [5, 15]           (options 0-1, labels 3-4)
    x3 costs: [0]               (only option, label 8)

Of smpmc-tiny's 8 combinations, exactly 3 satisfy the sketch's own property (P>=0.1[F "done"]) -- see
test_theory.py's TestColoredMdpTheoryOnFullAssignments for the general mechanism; here what matters is
their cost: (x1=1,x2=4)=25, (x1=4,x2=3)=45, (x1=4,x2=4)=55. Every property-satisfying combination not
listed here costs more than 55; every cheaper combination does not satisfy the property. That makes 25,
45 and 55 exact, meaningful boundaries to test against, rather than arbitrary numbers.
"""

from __future__ import annotations

from types import SimpleNamespace

import z3

import paynt.parameter_space.constraints
import paynt.parameter_space.parameter_space
import paynt.parameter_space.smt
import paynt.synthesizer.smpmc
import paynt.synthesizer.synthesizer

from tests.helpers.models import get_sketch_paths

_COSTS_FILE = get_sketch_paths("tests/smpmc-tiny", props_name="sketch.costs")[1]


def _run(colored_mdp, task, threshold, method="smpmc"):
    task.constraint_name = "costs"
    task.costs_threshold = threshold
    task.costs_file_path = _COSTS_FILE
    if method == "smpmc":
        return paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run()
    return paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()


class TestCostsConstraintOnSmpmc:
    def test_exact_cheapest_feasible_combination_is_found_at_its_own_cost(self, smpmc_tiny):
        result = _run(smpmc_tiny.colored_mdp, smpmc_tiny.task, threshold=25)
        assert result.success
        assert str(result.assignment) == "x1=1, x2=4, x3=8"

    def test_one_below_the_cheapest_feasible_cost_is_unsat(self, smpmc_tiny):
        result = _run(smpmc_tiny.colored_mdp, smpmc_tiny.task, threshold=24)
        assert not result.success

    def test_a_generous_threshold_is_sat(self, smpmc_tiny):
        result = _run(smpmc_tiny.colored_mdp, smpmc_tiny.task, threshold=100)
        assert result.success


class TestCostsConstraintOnCegis:
    """The same constraint, threaded through the other consumer -- must agree with SMPMC exactly, since both are querying the same underlying combinatorics,
    just via different search strategies."""

    def test_agrees_with_smpmc_at_the_exact_boundary(self, smpmc_tiny):
        result = _run(smpmc_tiny.factory.build(), smpmc_tiny.task, threshold=25, method="cegis")
        assert result.success
        assert str(result.assignment) == "x1=1, x2=4, x3=8"

    def test_agrees_with_smpmc_below_the_boundary(self, smpmc_tiny):
        result = _run(smpmc_tiny.factory.build(), smpmc_tiny.task, threshold=24, method="cegis")
        assert not result.success


def test_parameters_with_the_same_name_have_costs_of_their_own(tmp_path):
    """Two parameters h, option 1 costs 5: at most 5 allows option 1 for one of them."""
    space = paynt.parameter_space.parameter_space.ParameterSpace()
    for _ in range(2):
        space.add_parameter("h", ["0", "1"])
    (tmp_path / "sketch.costs").write_text("h 0 0\nh 1 5\nh 0 0\nh 1 5\n")
    task = SimpleNamespace(costs_file_path=str(tmp_path / "sketch.costs"), costs_threshold=5)
    solver = paynt.parameter_space.smt.SmtSolver(space, paynt.parameter_space.constraints.build_constraint("costs"), colored_mdp="unused", task=task)
    encoding = paynt.parameter_space.smt.ParameterSpaceEncoding(solver, space)
    allowed = set()
    while (assignment := encoding.pick_assignment()) is not None:
        options = tuple(assignment.parameter_options(parameter)[0] for parameter in range(2))
        allowed.add(options)
        solver.solver.add(z3.Not(z3.And([solver.solver_clauses[parameter][option] for parameter, option in enumerate(options)])))
    assert allowed == {(0, 0), (0, 1), (1, 0)}
