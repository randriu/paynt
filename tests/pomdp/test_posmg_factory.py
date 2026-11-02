import paynt.colored_mdp
import paynt.pomdp.posmg
import paynt.pomdp.posmg._utils
import paynt.synthesizer.search_node


class TestPosmgColoredMdpFactory:
    def test_factory_does_not_build_automatically(self, posmg_mec_test):
        """Guardrail for the "factories don't eagerly build" redesign: a freshly-constructed factory holds no colored_mdp/memory-size state of its own --
        build()/set_imperfect_memory_size() must be called explicitly to get one."""
        assert not hasattr(posmg_mec_test.factory, "colored_mdp")
        assert not hasattr(posmg_mec_test.factory, "current_memory_size")

    def test_load_sketch_produces_a_posmg_colored_mdp(self, posmg_mec_test):
        assert type(posmg_mec_test.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert posmg_mec_test.colored_mdp.feature_kind == "posmg"
        assert isinstance(posmg_mec_test.colored_mdp.feature_info, paynt.pomdp.posmg._utils.PosmgInfo)

    def test_build_produces_an_mdp(self, posmg_mec_test):
        node = paynt.synthesizer.search_node.SearchNode(posmg_mec_test.colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = posmg_mec_test.colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_create_smg_from_mdp_attaches_player_indications(self, posmg_mec_test):
        node = paynt.synthesizer.search_node.SearchNode(posmg_mec_test.colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = posmg_mec_test.colored_mdp.build(node.parameter_space)
        smg = paynt.pomdp.posmg._utils.create_smg_from_mdp(posmg_mec_test.colored_mdp.feature_info, node.mdp)
        assert smg.states == node.mdp.states

    def test_set_imperfect_memory_size_produces_a_fresh_colored_mdp(self, posmg_mec_test):
        reunfolded = posmg_mec_test.factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "posmg"
