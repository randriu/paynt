import pytest

import paynt.mdp_family
import paynt.parser.sketch

from tests.helpers.models import get_sketch_paths
from tests.helpers.reference import Reference


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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="one policy is certified for all members, although its action go is disabled for a=2")
def test_the_satisfied_members_are_those_of_the_reference(tmp_path):
    """The family of models/tests/prob-goal: with a=0 and a=1 a policy reaches the goal for sure, with a=2 none does."""
    sketch_path = get_sketch_paths("tests/prob-goal", "family.templ")[0]
    props_path = tmp_path / "goal.props"
    props_path.write_text('P>=0.5 [F "goal"]\n')
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, str(props_path))
    synthesizer = paynt.mdp_family.PolicyTreeSynthesizer(factory.build(), task)
    prop = task.get_property()
    evaluations = synthesizer.evaluate(prop=prop, print_stats=False)
    satisfied = set()
    for evaluation in evaluations:
        if evaluation.sat:
            assert synthesizer.verify_policy(evaluation.selected_choices, prop, evaluation.policy[0])
            space = evaluation.parameter_space
            satisfied |= {str(space.construct_assignment(combination)) for combination in space.all_combinations()}
    assert satisfied == {str(design) for design in Reference(sketch_path, ['P>=0.5 [F "goal"]']).feasible}
