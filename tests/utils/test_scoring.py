import pytest

import paynt.model.model
import paynt.utils.scoring
import paynt.synthesizer.search_node

from tests.helpers.models import load_sketch


@pytest.fixture
def colored_mdp_parameter_space_prop_result():
    """A plain, non-specialized ColoredMdp -- see tests/model/test_model_index.py's fixture docstring for why this no longer goes through
    paynt.quotient.quotient.Quotient (deleted)."""
    colored_mdp_factory, task = load_sketch("tests/generic-maze")
    colored_mdp = colored_mdp_factory.colored_mdp
    node = paynt.synthesizer.search_node.SearchNode(colored_mdp.parameter_space.copy())
    node.mdp, node.selected_choices = colored_mdp.build(node.parameter_space)
    prop = task.get_property()
    result = node.mdp.model_check_property(prop)
    return colored_mdp, node, prop, result


class TestSchedulerScoring:
    def test_estimate_scheduler_difference_returns_a_nonnegative_score_per_inconsistent_parameter(self, colored_mdp_parameter_space_prop_result):
        """
        estimate_scheduler_difference is a thin wrapper around a native payntbind call, so (unlike
        choice_values) there's no independent formula worth re-deriving here in Python -- this checks the
        contract instead: one score per inconsistent parameter, each non-negative (it's a weighted value
        difference). Real numerical correctness is exercised end-to-end by every AR-based specialist's CLI
        regression check, which all route splitting decisions through this exact function.
        """
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
    """
    Paper https://www.jair.org/index.php/jair/article/view/16593 Sec 3.3's L(h): cross-constraint
    disagreement among consistent candidate selections. Pure/hand-built -- no real model needed, since
    compute_incompatibility_levels only ever looks at option-index lists.
    """

    def test_fewer_than_two_candidates_yields_no_disagreement(self):
        assert paynt.utils.scoring.compute_incompatibility_levels([]) == {}
        assert paynt.utils.scoring.compute_incompatibility_levels([[[0], [1]]]) == {}

    def test_candidates_disagreeing_on_one_parameter_and_agreeing_on_another(self):
        # 3 candidates (one per still-undecided constraint), 2 parameters: they disagree on parameter 0
        # (three distinct values, in first-seen order) and agree on parameter 1 (always option 2)
        candidates = [[[0], [2]], [[1], [2]], [[3], [2]]]
        assert paynt.utils.scoring.compute_incompatibility_levels(candidates) == {0: [0, 1, 3]}

    def test_a_parameter_with_zero_options_in_some_candidate_contributes_no_opinion_there(self):
        """Unlike every other feature, "dt" does not pad an irrelevant parameter to one arbitrary option -- a candidate whose selection has an empty option list
        for some parameter must not crash and must not count as a vote."""
        candidates = [[[0], []], [[1], []], [[0], []]]
        assert paynt.utils.scoring.compute_incompatibility_levels(candidates) == {0: [0, 1]}

    def test_an_inconsistent_own_candidate_is_the_callers_responsibility_to_exclude(self):
        """compute_incompatibility_levels itself does not filter out locally-inconsistent selections (options of length > 1 for some parameter) -- callers
        (split_parameter_space / split_undecided_space) are responsible for only passing selections that are already fully consistent.

        Documented here since it's a real, easy-to-get-wrong contract, not asserted defensively in the function itself (hot path).
        """
        candidates = [[[0, 1]], [[2]]]
        # parameter 0's values across candidates are effectively {0, 1, 2} once fully expanded, but this
        # function only ever looks at singleton selections (len(options) == 1) when voting -- the first,
        # locally-inconsistent candidate is silently skipped rather than contributing multiple values
        assert paynt.utils.scoring.compute_incompatibility_levels(candidates) == {}

    def test_parameters_with_max_incompatibility_on_empty_input(self):
        assert paynt.utils.scoring.parameters_with_max_incompatibility({}) == []

    def test_parameters_with_max_incompatibility_on_unique_max(self):
        levels = {0: [1, 2], 1: [1, 2, 3]}
        assert paynt.utils.scoring.parameters_with_max_incompatibility(levels) == [1]

    def test_parameters_with_max_incompatibility_on_tied_max(self):
        levels = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8]}
        assert paynt.utils.scoring.parameters_with_max_incompatibility(levels) == [0, 2]
