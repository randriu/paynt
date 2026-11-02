import math

import pytest

import paynt.model.model
import paynt.synthesizer.search_node

from tests.helpers.models import load_sketch


@pytest.fixture
def colored_mdp_parameter_space_prop_result():
    """A plain, non-specialized ColoredMdp built directly from a PRISM DTMC-with-parameters sketch (no FSC/tree
    unfolding) -- this is the "generic" case paynt.quotient.quotient.Quotient used to handle, before it was
    deleted once its last two callers (SynthesizerHybrid.split, this exact sketch.py construction site)
    were migrated onto the shared ColoredMdp-based mechanisms."""
    colored_mdp_factory, task = load_sketch("tests/generic-maze")
    colored_mdp = colored_mdp_factory.colored_mdp
    node = paynt.synthesizer.search_node.SearchNode(colored_mdp.parameter_space.copy())
    node.mdp, node.selected_choices = colored_mdp.build(node.parameter_space)
    prop = task.get_property()
    result = node.mdp.model_check_property(prop)
    return colored_mdp, node, prop, result


class TestModelIndexScoring:
    def test_choice_values_matches_its_documented_formula(self, colored_mdp_parameter_space_prop_result):
        """Independent correctness check: hand-computes rew(c) + sum_s'[P(s,c,s')*mc(s')] directly from the raw transition matrix and reward model (the formula
        choice_values documents), for every choice, and compares against ModelIndex.choice_values's actual output.

        This used to compare against paynt.quotient.quotient.Quotient.choice_values instead (the method choice_values was extracted from) as an oracle, but that
        class no longer exists -- and re-deriving the formula independently is arguably the better test anyway, since a duplicate-implementation comparison
        can't catch a bug both sides share.
        """
        colored_mdp, node, prop, result = colored_mdp_parameter_space_prop_result
        state_values = result.result.get_value_vector()
        choice_values = paynt.model.model.ModelIndex.choice_values(node.mdp.model, prop, state_values)

        mdp = node.mdp.model
        tm = mdp.transition_matrix
        reward_model = mdp.reward_models.get(prop.formula.reward_name)
        choice_rewards = list(reward_model.state_action_rewards)
        for choice in range(mdp.nr_choices):
            expected = choice_rewards[choice]
            for entry in tm.get_row(choice):
                successor_value = state_values[entry.column]
                if successor_value == math.inf:
                    continue  # infinite successors are averaged in by make_vector_defined, not summed directly
                expected += entry.value * successor_value
            assert choice_values[choice] == pytest.approx(expected, abs=1e-6)

    def test_compute_expected_visits_is_nonnegative_and_visits_the_initial_state(self, colored_mdp_parameter_space_prop_result):
        colored_mdp, node, prop, result = colored_mdp_parameter_space_prop_result
        # choices must be local to node.mdp.model's own indexing, as scheduler_scores derives
        # them via result.get_scheduler().compute_action_support(...) -- not the underlying-model-global selected_choices
        local_choices = result.result.get_scheduler().compute_action_support(node.mdp.model.nondeterministic_choice_indices)
        visits = paynt.model.model.ModelIndex.compute_expected_visits(node.mdp.model, prop, local_choices)
        assert len(visits) == node.mdp.model.nr_states
        assert all(v >= 0 for v in visits)
        # the initial state is visited at least once by construction, before any transition happens
        initial_state = node.mdp.model.initial_states[0]
        assert visits[initial_state] >= 1

    def test_compute_expected_visits_respects_disable_flag(self, colored_mdp_parameter_space_prop_result):
        """Regression test: the original Quotient.compute_expected_visits returned a vector sized to the full underlying MDP's state count on this early-return
        path, which is wrong whenever `mdp` is a restricted sub-MDP with fewer states.

        ModelIndex sizes the vector to `mdp` itself instead. disable_expected_visits is a plain parameter (not a class attribute) precisely so two syntheses in
        the same process can't leak this setting into each other -- see paynt.task.SynthesisTask.
        """
        _, node, prop, result = colored_mdp_parameter_space_prop_result
        local_choices = result.result.get_scheduler().compute_action_support(node.mdp.model.nondeterministic_choice_indices)
        visits = paynt.model.model.ModelIndex.compute_expected_visits(node.mdp.model, prop, local_choices, disable_expected_visits=True)
        assert visits == [1] * node.mdp.model.nr_states

    def test_make_vector_defined_replaces_infinities_with_average_of_finite_values(self):
        vector = [1.0, math.inf, 3.0]
        result = paynt.model.model.ModelIndex.make_vector_defined(vector)
        assert result[0] == 1.0
        assert result[2] == 3.0
        assert result[1] == pytest.approx((1.0 + 0 + 3.0) / 3)
