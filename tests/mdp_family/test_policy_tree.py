import pytest

import paynt.mdp_family


class TestPolicyTreeSynthesis:
    @pytest.mark.parametrize("disable_expected_visits", [False, True])
    def test_every_family_member_gets_a_satisfying_policy(self, mdp_family_avoid_8_2_easy, disable_expected_visits):
        """Each policy is model-checked again with verify_policy.

        Without expected visits only the splitting heuristic changes, not the result.
        """
        task = mdp_family_avoid_8_2_easy.task
        task.disable_expected_visits = disable_expected_visits
        synthesizer = paynt.mdp_family.PolicyTreeSynthesizer(mdp_family_avoid_8_2_easy.colored_mdp, task)
        prop = task.get_property()
        evaluations = synthesizer.evaluate(prop=prop, print_stats=False)
        # 3 policies: solver output, a regression value
        assert len(evaluations) == 3
        assert all(evaluation.sat for evaluation in evaluations)
        for evaluation in evaluations:
            assert synthesizer.verify_policy(evaluation.selected_choices, prop, evaluation.policy[0])
