import paynt.colored_mdp
import paynt.pomdp
import paynt.pomdp._utils


class TestPomdpColoredMdpFactory:
    def test_factory_does_not_build_automatically(self, pomdp_maze):
        """A factory builds a colored MDP, and unfolds the memory, only when asked."""
        assert not hasattr(pomdp_maze.factory, "colored_mdp")
        assert not hasattr(pomdp_maze.factory, "current_memory_size")

    def test_load_sketch_produces_a_pomdp_colored_mdp(self, pomdp_maze):
        assert type(pomdp_maze.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert pomdp_maze.colored_mdp.feature_kind == "pomdp"
        assert isinstance(pomdp_maze.colored_mdp.feature_info, paynt.pomdp._utils.PomdpInfo)

    def test_more_memory_gives_more_parameters(self, pomdp_maze):
        reunfolded = pomdp_maze.factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "pomdp"
        assert reunfolded.parameter_space.num_parameters > pomdp_maze.colored_mdp.parameter_space.num_parameters
