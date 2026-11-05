import re

import paynt.colored_mdp
import paynt.pomdp


def agents_with_a_second_memory_node(colored_mdp):
    """The agents that have an action parameter A(agent, observation, 1)."""
    space = colored_mdp.parameter_space
    names = [str(space.parameter_name(parameter)) for parameter in range(space.num_parameters)]
    return {match.group(1) for match in map(re.compile(r"A\((\d+),.*,1\)").fullmatch, names) if match}


class TestDecPomdpColoredMdpFactory:
    def test_load_sketch_produces_a_decpomdp_colored_mdp(self, decpomdp_dectiger):
        assert type(decpomdp_dectiger.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert decpomdp_dectiger.colored_mdp.feature_kind == "decpomdp"
        assert decpomdp_dectiger.colored_mdp.feature_info is None

    def test_factory_reports_agent_count(self, decpomdp_dectiger):
        assert isinstance(decpomdp_dectiger.factory, paynt.pomdp.decpomdp.DecPomdpColoredMdpFactory)
        assert decpomdp_dectiger.factory.nr_agents == 2

    def test_more_memory_gives_every_agent_a_second_memory_node(self, decpomdp_dectiger):
        reunfolded = decpomdp_dectiger.factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "decpomdp"
        assert agents_with_a_second_memory_node(reunfolded) == {"0", "1"}

    def test_more_memory_for_one_agent_leaves_the_other_alone(self, decpomdp_dectiger):
        """It changes the memory of one agent in what build() set up for all of them."""
        decpomdp_dectiger.factory.build()
        reunfolded = decpomdp_dectiger.factory.set_agent_imperfect_memory_size(0, 2)
        assert agents_with_a_second_memory_node(reunfolded) == {"0"}
