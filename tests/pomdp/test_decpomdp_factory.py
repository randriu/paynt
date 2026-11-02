import paynt.colored_mdp
import paynt.pomdp
import paynt.synthesizer.search_node


class TestDecPomdpColoredMdpFactory:
    def test_factory_does_not_build_automatically(self, decpomdp_dectiger):
        """Guardrail for the "factories don't eagerly build" redesign: a freshly-constructed factory holds no colored_mdp/memory-size state of its own --
        build()/set_imperfect_memory_size() must be called explicitly to get one."""
        assert not hasattr(decpomdp_dectiger.factory, "colored_mdp")
        assert not hasattr(decpomdp_dectiger.factory, "current_memory_size")

    def test_load_sketch_produces_a_decpomdp_colored_mdp(self, decpomdp_dectiger):
        assert type(decpomdp_dectiger.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert decpomdp_dectiger.colored_mdp.feature_kind == "decpomdp"
        assert decpomdp_dectiger.colored_mdp.feature_info is None

    def test_factory_reports_agent_count(self, decpomdp_dectiger):
        assert isinstance(decpomdp_dectiger.factory, paynt.pomdp.decpomdp.DecPomdpColoredMdpFactory)
        assert decpomdp_dectiger.factory.nr_agents == 2

    def test_build_produces_an_mdp(self, decpomdp_dectiger):
        node = paynt.synthesizer.search_node.SearchNode(decpomdp_dectiger.colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = decpomdp_dectiger.colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_set_imperfect_memory_size_produces_a_fresh_colored_mdp(self, decpomdp_dectiger):
        reunfolded = decpomdp_dectiger.factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "decpomdp"

    def test_set_agent_imperfect_memory_size_produces_a_fresh_colored_mdp(self, decpomdp_dectiger):
        """Per-agent memory sizing: distinct from set_imperfect_memory_size, which resizes every agent.

        This is a partial update over an existing per-agent baseline, not a standalone first build -- build() establishes that baseline for every agent first,
        matching how any real caller would need to use it.
        """
        decpomdp_dectiger.factory.build()
        reunfolded = decpomdp_dectiger.factory.set_agent_imperfect_memory_size(0, 2)
        assert reunfolded.feature_kind == "decpomdp"
