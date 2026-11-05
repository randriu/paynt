import paynt.colored_mdp
import paynt.pomdp.posmg
import paynt.pomdp.posmg._utils
import paynt.synthesizer.search_node


class TestPosmgColoredMdpFactory:
    def test_load_sketch_produces_a_posmg_colored_mdp(self, posmg_mec_test):
        assert type(posmg_mec_test.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert posmg_mec_test.colored_mdp.feature_kind == "posmg"
        assert isinstance(posmg_mec_test.colored_mdp.feature_info, paynt.pomdp.posmg._utils.PosmgInfo)

    def test_create_smg_from_mdp_attaches_player_indications(self, posmg_mec_test):
        node = paynt.synthesizer.search_node.SearchNode(posmg_mec_test.colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = posmg_mec_test.colored_mdp.build(node.parameter_space)
        smg = paynt.pomdp.posmg._utils.create_smg_from_mdp(posmg_mec_test.colored_mdp.feature_info, node.mdp)
        assert smg.states == node.mdp.states
