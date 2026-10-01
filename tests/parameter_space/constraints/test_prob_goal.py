"""Integration tests for ProbGoalConstraint against models/tests/smpmc-tiny, whose 8 combinations happen to give reachability probability of exactly 0.0 or 1.0
-- never anything in between -- for reaching "done" (verified directly against Storm: only (x1=1,x2=4), (x1=4,x2=3) and (x1=4,x2=4), all with x3=8, give 1.0;
every other combination gives 0.0).

That makes it an unambiguous fixture for prob0/prob1: since every parameter is fully determined for a "generic" sketch (no residual nondeterminism, see
checker.py's singleton-eta shortcut), "possible" (prob0) and "almost-sure" (prob1) coincide here on exactly the same 3-combination set -- the constraint should
accept only those and refute everything else.
"""

from __future__ import annotations

import paynt.synthesizer.smpmc
import paynt.synthesizer.synthesizer

_GOOD_ASSIGNMENTS = {"x1=1, x2=4, x3=8", "x1=4, x2=3, x3=8", "x1=4, x2=4, x3=8"}


class TestProbGoalConstraintOnSmpmc:
    def test_prob1_finds_an_almost_surely_reaching_assignment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        smpmc_tiny_task.constraint_name = "prob1"
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success
        assert str(result.assignment) in _GOOD_ASSIGNMENTS

    def test_prob0_finds_a_possibly_reaching_assignment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        smpmc_tiny_task.constraint_name = "prob0"
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success
        assert str(result.assignment) in _GOOD_ASSIGNMENTS

    def test_prob1_is_unsat_when_every_almost_surely_reaching_option_is_excluded(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        # x1's options are indexed [0,1,2,3] for labels [1,2,3,4]; every reaching assignment needs
        # x1 in {1,4} (indices 0 or 3), so narrowing x1 to {2,3} (indices 1,2) excludes all of them
        smpmc_tiny_colored_mdp.parameter_space.parameter_set_options(0, [1, 2])
        smpmc_tiny_task.constraint_name = "prob1"
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert not result.success


class TestProbGoalConstraintOnCegis:
    def test_agrees_with_smpmc_on_prob1(self, smpmc_tiny_colored_mdp_factory, smpmc_tiny_task):
        smpmc_tiny_task.constraint_name = "prob1"
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(smpmc_tiny_colored_mdp_factory.build(), smpmc_tiny_task, "cegis").run()
        assert result.success
        assert str(result.assignment) in _GOOD_ASSIGNMENTS
