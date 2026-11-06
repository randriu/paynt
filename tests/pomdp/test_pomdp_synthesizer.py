import json
import re

import pytest
import stormpy

import paynt.pomdp
import paynt.pomdp._utils
import paynt.pomdp.fsc


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

    def test_the_fsc_attains_the_synthesized_value(self, pomdp_maze):
        """The Markov chain the FSC induces in the POMDP, model-checked on its own; with one memory node and 8 observations, the FSC has 8 actions and 8
        memory updates: a size of 16."""
        synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        fsc = paynt.pomdp._utils.assignment_to_fsc(synthesizer.colored_mdp.feature_info, assignment)
        assert fsc.num_nodes == 1
        assert paynt.pomdp._utils.policy_size(synthesizer.colored_mdp, assignment) == 16
        dtmc = paynt.pomdp._utils.get_induced_dtmc_from_fsc(synthesizer.colored_mdp.feature_info, fsc)
        value = stormpy.model_checking(dtmc, synthesizer.task.get_property().formula).get_value(dtmc.initial_states[0])
        assert value == pytest.approx(synthesizer.task.specification.optimality.optimum, rel=1e-4)

    @pytest.mark.xfail(strict=True, reason="from_json drops the labels and makes a deterministic FSC stochastic")
    def test_an_fsc_read_back_from_json_is_the_same(self, pomdp_maze):
        synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        fsc = paynt.pomdp._utils.assignment_to_fsc(synthesizer.colored_mdp.feature_info, assignment)
        written = json.loads(json.dumps(fsc.to_json()))
        read_back = paynt.pomdp.fsc.FscFactored.from_json(written)
        assert read_back.to_json() == written
        assert read_back.is_deterministic == fsc.is_deterministic
