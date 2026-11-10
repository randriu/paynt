import pytest

import paynt.model.model
import paynt.utils.scoring
import paynt.synthesizer.search_node

from tests.helpers.models import load_sketch


@pytest.fixture
def colored_mdp_parameter_space_prop_result():
    """The colored MDP of a DTMC sketch, its whole space as one node, the property and the model checking result of the node's MDP."""
    colored_mdp_factory, task = load_sketch("tests/generic-maze")
    colored_mdp = colored_mdp_factory.colored_mdp
    node = paynt.synthesizer.search_node.SearchNode(colored_mdp.parameter_space.copy())
    node.mdp, node.selected_choices = colored_mdp.build(node.parameter_space)
    prop = task.get_property()
    result = node.mdp.model_check_property(prop)
    return colored_mdp, node, prop, result


class TestSchedulerScoring:
    def test_estimate_scheduler_difference_returns_a_nonnegative_score_per_inconsistent_parameter(self, colored_mdp_parameter_space_prop_result):
        """The scores come from payntbind, so only their shape is checked: one score per inconsistent parameter, each a non-negative weighted difference."""
        colored_mdp, node, prop, result = colored_mdp_parameter_space_prop_result
        selection = colored_mdp.scheduler_selection(node.mdp, result.result.get_scheduler())
        inconsistent_assignments = {parameter: options for parameter, options in enumerate(selection) if len(options) > 1}
        assert inconsistent_assignments, "expected at least one inconsistent parameter for this fixture"

        choice_values = paynt.model.model.ModelIndex.choice_values(node.mdp.model, prop, result.result.get_value_vector())
        local_choices = result.result.get_scheduler().compute_action_support(node.mdp.model.nondeterministic_choice_indices)
        expected_visits = paynt.model.model.ModelIndex.compute_expected_visits(node.mdp.model, prop, local_choices)
        underlying_choice_map = list(range(node.mdp.model.nr_choices))

        scores = paynt.utils.scoring.estimate_scheduler_difference(
            colored_mdp, node.mdp.model, underlying_choice_map, inconsistent_assignments, choice_values, expected_visits
        )

        assert set(scores.keys()) == set(inconsistent_assignments.keys())
        assert all(score >= 0 for score in scores.values())

    def test_parameters_with_max_score(self):
        score = {0: 5, 1: 3, 2: 5}
        assert paynt.utils.scoring.parameters_with_max_score(score) == [0, 2]


class TestIncompatibilityScoring:
    """L(h) of https://www.jair.org/index.php/jair/article/view/16593, Section 3.3: the options on which the selections of the undecided properties
    disagree."""

    def test_fewer_than_two_candidates_yields_no_disagreement(self):
        assert paynt.utils.scoring.compute_incompatibility_levels([]) == {}
        assert paynt.utils.scoring.compute_incompatibility_levels([[[0], [1]]]) == {}

    def test_candidates_disagreeing_on_one_parameter_and_agreeing_on_another(self):
        # 3 candidates (one per still-undecided constraint), 2 parameters: they disagree on parameter 0
        # (three distinct values, in first-seen order) and agree on parameter 1 (always option 2)
        candidates = [[[0], [2]], [[1], [2]], [[3], [2]]]
        assert paynt.utils.scoring.compute_incompatibility_levels(candidates) == {0: [0, 1, 3]}

    def test_a_parameter_with_zero_options_in_some_candidate_contributes_no_opinion_there(self):
        """Decision trees leave an irrelevant parameter without options: that is no vote."""
        candidates = [[[0], []], [[1], []], [[0], []]]
        assert paynt.utils.scoring.compute_incompatibility_levels(candidates) == {0: [0, 1]}

    def test_an_inconsistent_own_candidate_is_the_callers_responsibility_to_exclude(self):
        """Only a single option is a vote, so an inconsistent selection does not count; the callers pass consistent ones only."""
        candidates = [[[0, 1]], [[2]]]
        assert paynt.utils.scoring.compute_incompatibility_levels(candidates) == {}

    def test_parameters_with_max_incompatibility_on_empty_input(self):
        assert paynt.utils.scoring.parameters_with_max_incompatibility({}) == []

    def test_parameters_with_max_incompatibility_on_unique_max(self):
        levels = {0: [1, 2], 1: [1, 2, 3]}
        assert paynt.utils.scoring.parameters_with_max_incompatibility(levels) == [1]

    def test_parameters_with_max_incompatibility_on_tied_max(self):
        levels = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8]}
        assert paynt.utils.scoring.parameters_with_max_incompatibility(levels) == [0, 2]
