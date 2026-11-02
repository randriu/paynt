import paynt.colored_mdp
import paynt.mdp_family
import paynt.mdp_family._utils


class TestFamilyColoredMdpFactory:
    def test_factory_does_not_build_automatically(self, mdp_family_avoid_8_2_easy):
        """A factory builds its colored MDP only when build() is called."""
        assert not hasattr(mdp_family_avoid_8_2_easy.factory, "colored_mdp")

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
