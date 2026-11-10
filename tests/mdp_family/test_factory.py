import pytest

import paynt.colored_mdp
import paynt.mdp_family
import paynt.mdp_family._utils
import paynt.parser.sketch

from tests.helpers.models import get_sketch_paths


class TestFamilyColoredMdpFactory:
    def test_load_sketch_produces_a_family_colored_mdp(self, mdp_family_avoid_8_2_easy):
        assert type(mdp_family_avoid_8_2_easy.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert mdp_family_avoid_8_2_easy.colored_mdp.feature_kind == "family"
        assert isinstance(mdp_family_avoid_8_2_easy.colored_mdp.feature_info, paynt.mdp_family._utils.MdpFamilyInfo)

    def test_build_assignment_does_not_force_a_dtmc(self, mdp_family_avoid_8_2_easy):
        """A member of the family is an MDP: fixing the environment does not fix the agent's policy."""
        assignment = mdp_family_avoid_8_2_easy.colored_mdp.parameter_space.pick_any()
        assert not mdp_family_avoid_8_2_easy.colored_mdp.build_assignment(assignment).is_deterministic

    def test_empty_policy_has_one_entry_per_state(self, mdp_family_avoid_8_2_easy):
        policy = paynt.mdp_family._utils.empty_policy(mdp_family_avoid_8_2_easy.colored_mdp)
        assert len(policy) == mdp_family_avoid_8_2_easy.colored_mdp.underlying_mdp.nr_states
        assert all(action is None for action in policy)

    def test_action_structure_is_populated(self, mdp_family_avoid_8_2_easy):
        info = mdp_family_avoid_8_2_easy.colored_mdp.feature_info
        assert info.num_actions > 0
        assert len(info.action_labels) == info.num_actions
        assert len(info.state_to_actions) == mdp_family_avoid_8_2_easy.colored_mdp.underlying_mdp.nr_states


@pytest.mark.parametrize("prop", ['Pmax=? [F "goal"]', 'Pmin=? [F "goal"]'])
def test_memory_unfolded_into_a_family_keeps_the_value_of_every_member(tmp_path, prop):
    """The family of models/tests/prob-goal: memoryless policies are as good as any for reaching the goal."""
    sketch_path = get_sketch_paths("tests/prob-goal", "family.templ")[0]
    (tmp_path / "goal.props").write_text(prop + "\n")
    values = []
    for memory_size in (1, 2):
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, str(tmp_path / "goal.props"), task_kwargs={"memory_size": memory_size})
        colored_mdp = factory.build()
        assert colored_mdp.underlying_mdp.nr_states == 3 * memory_size
        space = colored_mdp.parameter_space
        members = [space.construct_assignment(combination) for combination in space.all_combinations()]
        values.append([colored_mdp.build_assignment(member).model_check_property(task.get_property()).value for member in members])
    assert values[1] == pytest.approx(values[0])
    assert values[0] == pytest.approx([1, 1, 0] if "max" in prop else [1, 0, 0])
