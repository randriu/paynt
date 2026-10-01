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

import paynt.synthesizer.smpmc
import paynt.synthesizer.synthesizer

from helpers.helper import get_sketch_paths

_COSTS_FILE = get_sketch_paths("tests/smpmc-tiny", props_name="sketch.costs")[1]


def _run(colored_mdp, task, threshold, method="smpmc"):
    task.constraint_name = "costs"
    task.costs_threshold = threshold
    task.costs_file_path = _COSTS_FILE
    if method == "smpmc":
        return paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run()
    return paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()


class TestCostsConstraintOnSmpmc:
    def test_exact_cheapest_feasible_combination_is_found_at_its_own_cost(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = _run(smpmc_tiny_colored_mdp, smpmc_tiny_task, threshold=25)
        assert result.success
        assert str(result.assignment) == "x1=1, x2=4, x3=8"

    def test_one_below_the_cheapest_feasible_cost_is_unsat(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = _run(smpmc_tiny_colored_mdp, smpmc_tiny_task, threshold=24)
        assert not result.success

    def test_a_generous_threshold_is_sat(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = _run(smpmc_tiny_colored_mdp, smpmc_tiny_task, threshold=100)
        assert result.success


class TestCostsConstraintOnCegis:
    """The same constraint, threaded through the other consumer -- must agree with SMPMC exactly, since both are querying the same underlying combinatorics,
    just via different search strategies."""

    def test_agrees_with_smpmc_at_the_exact_boundary(self, smpmc_tiny_colored_mdp_factory, smpmc_tiny_task):
        result = _run(smpmc_tiny_colored_mdp_factory.build(), smpmc_tiny_task, threshold=25, method="cegis")
        assert result.success
        assert str(result.assignment) == "x1=1, x2=4, x3=8"

    def test_agrees_with_smpmc_below_the_boundary(self, smpmc_tiny_colored_mdp_factory, smpmc_tiny_task):
        result = _run(smpmc_tiny_colored_mdp_factory.build(), smpmc_tiny_task, threshold=24, method="cegis")
        assert not result.success
