import re

import pytest

import paynt.pomdp
import paynt.pomdp._utils


class TestPomdpSynthesis:
    def test_synthesize_converges_to_the_known_optimum(self, pomdp_maze):
        """The best FSC with one memory node; AR proves it optimal, but the value and the actions are solver output, regression values."""
        synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        optimum = synthesizer.task.specification.optimality.optimum
        assert optimum == pytest.approx(71.92884615384588, abs=1e-6)
        # only the actions: the observation labels are printed with tabs
        chosen_actions = sorted(re.findall(r"=(\w+)", str(assignment)))
        assert chosen_actions == ["down", "left", "right", "right", "up", "up", "up"]

    def test_assignment_to_fsc_and_policy_size_do_not_crash(self, pomdp_maze):
        """With one memory node and 8 observations, the FSC has 8 actions and 8 memory updates: a size of 16."""
        synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        fsc = paynt.pomdp._utils.assignment_to_fsc(synthesizer.colored_mdp.feature_info, assignment)
        assert fsc.num_nodes == 1
        assert paynt.pomdp._utils.policy_size(synthesizer.colored_mdp, assignment) == 16
