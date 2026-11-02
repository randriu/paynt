import re

import pytest

import paynt.pomdp
import paynt.pomdp._utils


class TestPomdpSynthesis:
    def test_synthesize_converges_to_the_known_optimum(self, pomdp_maze):
        """Regression test for the full AR + POMDP-specialized scoring path: this sketch's parameter space (1e10) is small enough that AR fully converges within
        the test, so the reported optimum is the true value for a 1-node FSC (the fixture's default memory size), not just a snapshot -- and AR's correctness
        doesn't depend on the splitting heuristic (utils.scoring.estimate_scheduler_difference_pomdp) being right, only on it terminating, so this also catches
        that free function crashing or looping.

        Since synthesis is deterministic, the actions the assignment picks are checked too (not just the reward value) -- the observation labels embedded in
        str(assignment) contain incidental tab formatting from stormpy's own label rendering, so only the "=action" side is extracted, avoiding a brittle
        whitespace-sensitive comparison while still checking the synthesized decision itself.
        """
        synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        optimum = synthesizer.task.specification.optimality.optimum
        assert optimum == pytest.approx(71.92884615384588, abs=1e-6)
        chosen_actions = sorted(re.findall(r"=(\w+)", str(assignment)))
        assert chosen_actions == ["down", "left", "right", "right", "up", "up", "up"]

    def test_assignment_to_fsc_and_policy_size_do_not_crash(self, pomdp_maze):
        """Regression test for the paynt.quotient.fsc.FSC -> FscFactored typo bug found during migration:

        assignment_to_fsc used to raise AttributeError (no such name in the fsc module) whenever actually
        called -- previously unreachable except via SAYNT, so nothing had caught it.
        """
        synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        fsc = paynt.pomdp._utils.assignment_to_fsc(synthesizer.colored_mdp.feature_info, assignment)
        assert fsc.num_nodes == 1
        assert paynt.pomdp._utils.policy_size(synthesizer.colored_mdp, assignment) == 16
