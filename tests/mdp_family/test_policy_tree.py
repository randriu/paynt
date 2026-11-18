import pytest

import paynt.mdp_family
import paynt.mdp_family._utils
import paynt.parser.sketch

from tests.helpers.models import get_sketch_paths
from tests.helpers.reference import Reference
from tests.mdp_family.test_random_families import satisfies


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
            assert synthesizer.verify_policy(evaluation.parameter_space, evaluation.selected_choices, prop, evaluation.policy[0])


def load_family(project, sketch_name, prop_text, tmp_path):
    props_path = tmp_path / "goal.props"
    props_path.write_text(prop_text + "\n")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths(project, sketch_name)[0], str(props_path))
    return factory.build(), task


def test_a_game_cut_off_reports_it(tmp_path):
    """The game of prob-goal needs two policy iterations."""
    colored_mdp, task = load_family("tests/prob-goal", "family.templ", 'P>=0.5 [F "goal"]', tmp_path)
    solver = paynt.mdp_family._utils.build_game_abstraction_solver(colored_mdp, task.get_property())
    _, choices = colored_mdp.build(colored_mdp.parameter_space)
    stuck = paynt.mdp_family._utils.stuck_choices(colored_mdp, colored_mdp.parameter_space)
    assert solver.max_policy_iterations == 100
    solver.solve_sg(choices, stuck)
    assert solver.solution_converged
    solver.max_policy_iterations = 1
    solver.solve_sg(choices, stuck)
    assert not solver.solution_converged


@pytest.mark.parametrize("max_policy_iterations", [None, 1])
@pytest.mark.parametrize(
    "project, sketch_name, prop_text",
    [
        ("tests/prob-goal", "family.templ", 'P>=0.5 [F "goal"]'),
        ("tests/edge-cases", "policy-tree-merge.templ", 'P<=1.1 [F "goal"]'),
        ("tests/edge-cases", "policy-tree-game-cycle.templ", 'P>=0.708333 [F "goal"]'),
    ],
)
def test_the_satisfied_members_are_those_of_the_reference(project, sketch_name, prop_text, max_policy_iterations, tmp_path, monkeypatch):
    """Each sketch's comment says what it tests.

    With one policy iteration most games are cut off and decide nothing.
    """
    if max_policy_iterations is not None:
        build = paynt.mdp_family._utils.build_game_abstraction_solver

        def build_with_cap(*args):
            solver = build(*args)
            solver.max_policy_iterations = max_policy_iterations
            return solver

        monkeypatch.setattr(paynt.mdp_family._utils, "build_game_abstraction_solver", build_with_cap)
    sketch_path = get_sketch_paths(project, sketch_name)[0]
    colored_mdp, task = load_family(project, sketch_name, prop_text, tmp_path)
    synthesizer = paynt.mdp_family.PolicyTreeSynthesizer(colored_mdp, task)
    prop = task.get_property()
    satisfied = set()
    for evaluation in synthesizer.evaluate(prop=prop, print_stats=False):
        if not evaluation.sat:
            continue
        space = evaluation.parameter_space
        for combination in space.all_combinations():
            member = space.construct_assignment(combination)
            assert satisfies(colored_mdp, prop, member, evaluation.policy[0]), str(member)
            satisfied.add(str(member))
    assert satisfied == {str(design) for design in Reference(sketch_path, [prop_text]).feasible}
