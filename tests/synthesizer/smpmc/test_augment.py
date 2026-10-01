"""Integration tests of add_policy_parameters on models/tests/mdp-family-rocks-4-2: a family of 81 environments (4 parameters) over an MDP with 162 states, 48
of which have more than one action."""

from __future__ import annotations

import pytest

import paynt.parser.sketch
import paynt.synthesizer.smpmc._augment


class TestAddPolicyParameters:
    def test_adds_one_policy_parameter_per_state_with_a_choice(self, rocks_colored_mdp):
        augmented, environment_parameters = paynt.synthesizer.smpmc._augment.add_policy_parameters(rocks_colored_mdp)
        assert environment_parameters == [0, 1, 2, 3]
        assert augmented.parameter_space.num_parameters == 4 + 48
        names = [augmented.parameter_space.parameter_name(parameter) for parameter in range(augmented.parameter_space.num_parameters)]
        assert names[:4] == ["o1x", "o1y", "o2x", "o2y"]
        # molehill's naming: the state valuation, without prism2jani's location variables
        assert "A([clk=0&visit1=0&visit2=0&x=1&y=1])" in names

    def test_policy_options_are_the_state_actions(self, rocks_colored_mdp):
        augmented, _ = paynt.synthesizer.smpmc._augment.add_policy_parameters(rocks_colored_mdp)
        info = rocks_colored_mdp.feature_info
        nci = rocks_colored_mdp.underlying_mdp.nondeterministic_choice_indices
        choice_to_assignment = augmented.coloring.getChoiceToAssignment()
        parameter = 4
        for state in range(rocks_colored_mdp.underlying_mdp.nr_states):
            actions = info.state_to_actions[state]
            if len(actions) < 2:
                continue
            assert augmented.parameter_space.parameter_to_option_labels[parameter] == [info.action_labels[action] for action in actions]
            for choice in range(nci[state], nci[state + 1]):
                assert (parameter, actions.index(info.choice_to_action[choice])) in choice_to_assignment[choice]
            parameter += 1

    def test_a_full_assignment_induces_a_dtmc(self, rocks_colored_mdp):
        augmented, _ = paynt.synthesizer.smpmc._augment.add_policy_parameters(rocks_colored_mdp)
        model = augmented.build_assignment(augmented.parameter_space.pick_any())
        assert model.model.nr_choices == model.model.nr_states

    def test_fixing_only_the_environment_leaves_the_policy_open(self, rocks_colored_mdp):
        augmented, environment_parameters = paynt.synthesizer.smpmc._augment.add_policy_parameters(rocks_colored_mdp)
        environment = augmented.parameter_space.copy()
        for parameter in environment_parameters:
            environment.parameter_set_options(parameter, environment.parameter_options(parameter)[:1])
        mdp, _ = augmented.build(environment)
        assert mdp.model.nr_choices > mdp.model.nr_states

    def test_the_family_itself_is_left_untouched(self, rocks_colored_mdp):
        """ParameterSpace.copy() shares its name/label lists, so a careless augmentation would grow the family's too."""
        paynt.synthesizer.smpmc._augment.add_policy_parameters(rocks_colored_mdp)
        parameter_space = rocks_colored_mdp.parameter_space
        assert parameter_space.num_parameters == 4
        assert len(parameter_space.parameter_to_name) == 4
        assert len(parameter_space.parameter_to_option_labels) == 4

    def test_rejects_an_environment_enabling_two_choices_of_one_action(self, tmp_path):
        """A policy picks actions: were two go-choices enabled at once (here for h1=0, h2=0), neither the policy nor the environment would pick among them."""
        (tmp_path / "sketch.templ").write_text(
            "mdp\n"
            "hole int h1 in {0..1};\n"
            "hole int h2 in {0..1};\n"
            "module m\n"
            "  s : [0..2] init 0;\n"
            "  [go] s=0 & h1=0 -> (s'=1);\n"
            "  [go] s=0 & h2=0 -> (s'=2);\n"
            "  [stay] s=0 -> true;\n"
            "  [done] s>0 -> true;\n"
            "endmodule\n"
            'label "goal" = s=1;\n'
        )
        (tmp_path / "sketch.props").write_text('P>=0.5 [F "goal"]\n')
        factory, _task = paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))
        with pytest.raises(ValueError, match="several choices of action 'go'"):
            paynt.synthesizer.smpmc._augment.add_policy_parameters(factory.build())

    def test_rejects_an_environment_enabling_two_identically_colored_choices_of_one_action(self, tmp_path):
        """The same environment turns on both: neither the policy nor the environment would pick among them."""
        (tmp_path / "sketch.templ").write_text(
            "mdp\n"
            "hole int h in {0..1};\n"
            "module m\n"
            "  s : [0..2] init 0;\n"
            "  [go] s=0 & h=0 -> (s'=1);\n"
            "  [go] s=0 & h=0 -> (s'=2);\n"
            "  [stay] s=0 -> true;\n"
            "  [done] s>0 -> true;\n"
            "endmodule\n"
            'label "goal" = s=1;\n'
        )
        (tmp_path / "sketch.props").write_text('P>=0.5 [F "goal"]\n')
        factory, _task = paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))
        with pytest.raises(ValueError, match="several choices of action 'go'"):
            paynt.synthesizer.smpmc._augment.add_policy_parameters(factory.build())

    def test_two_choices_of_one_action_for_different_values_of_one_parameter_are_fine(self, tmp_path):
        """What a family is for: the environment picks which choice the action stands for."""
        (tmp_path / "sketch.templ").write_text(
            "mdp\n"
            "hole int h in {0..1};\n"
            "module m\n"
            "  s : [0..2] init 0;\n"
            "  [go] s=0 & h=0 -> (s'=1);\n"
            "  [go] s=0 & h=1 -> (s'=2);\n"
            "  [stay] s=0 -> true;\n"
            "  [done] s>0 -> true;\n"
            "endmodule\n"
            'label "goal" = s=1;\n'
        )
        (tmp_path / "sketch.props").write_text('P>=0.5 [F "goal"]\n')
        factory, _task = paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))
        augmented, environment_parameters = paynt.synthesizer.smpmc._augment.add_policy_parameters(factory.build())
        assert environment_parameters == [0]
        assert augmented.parameter_space.num_parameters == 2  # the policy of state 0, choosing between go and stay

    def test_rejects_an_action_enabled_in_some_environments_only(self, tmp_path):
        """A policy picking a in state 0 would deadlock for h=0."""
        (tmp_path / "sketch.templ").write_text(
            "mdp\n"
            "hole int h in {0..1};\n"
            "module m\n"
            "  s : [0..2] init 0;\n"
            "  [a] s=0 & h=1 -> (s'=1);\n"
            "  [b] s=0 -> (s'=2);\n"
            "  [done] s>0 -> true;\n"
            "endmodule\n"
            'label "goal" = s=1;\n'
        )
        (tmp_path / "sketch.props").write_text('P>=0.5 [F "goal"]\n')
        factory, _task = paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))
        with pytest.raises(ValueError, match="action 'a' is enabled in some environments only"):
            paynt.synthesizer.smpmc._augment.add_policy_parameters(factory.build())

    def test_only_a_family_can_be_augmented(self, smpmc_tiny_colored_mdp):
        with pytest.raises(AssertionError):
            paynt.synthesizer.smpmc._augment.add_policy_parameters(smpmc_tiny_colored_mdp)


class TestStateNames:
    """The names of the policy parameters: a state's valuation, or its index for a model without any (a DRN file, say)."""

    def test_a_model_without_state_valuations_names_its_states_by_index(self):
        class WithoutValuations:
            nr_states = 3

            @staticmethod
            def has_state_valuations():
                return False

        assert paynt.synthesizer.smpmc._augment._state_names(WithoutValuations()) == ["s0", "s1", "s2"]

    def test_a_model_with_valuations_names_them_without_the_location_variables(self, rocks_colored_mdp):
        names = paynt.synthesizer.smpmc._augment._state_names(rocks_colored_mdp.underlying_mdp)
        assert len(names) == rocks_colored_mdp.underlying_mdp.nr_states
        assert "clk=0&visit1=0&visit2=0&x=1&y=1" in names
        assert not any("_loc_prism2jani" in name for name in names)
