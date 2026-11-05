import pytest

import paynt.colored_mdp
import paynt.parser.sketch
import paynt.pomdp
import paynt.pomdp._utils

# o is 0 in the initial state and 1 everywhere else
OBSERVABLE = """pomdp

observables
    o
endobservables

module m
    s : [0..2] init 0;
    o : [0..1] init 0;
    [a] s=0 -> 0.5 : (s'=1) & (o'=1) + 0.5 : (s'=2) & (o'=1);
    [b] s=0 -> (s'=2) & (o'=1);
    [a] s>0 -> true;
    [b] s>0 -> true;
endmodule

label "goal" = s=1;
"""


def observable_pomdp(tmp_path):
    (tmp_path / "sketch.templ").write_text(OBSERVABLE)
    (tmp_path / "sketch.props").write_text('Pmax=? [F "goal"]\n')
    return paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))[0]


class TestPomdpColoredMdpFactory:
    def test_load_sketch_produces_a_pomdp_colored_mdp(self, pomdp_maze):
        assert type(pomdp_maze.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert pomdp_maze.colored_mdp.feature_kind == "pomdp"
        assert isinstance(pomdp_maze.colored_mdp.feature_info, paynt.pomdp._utils.PomdpInfo)

    def test_more_memory_gives_more_parameters(self, pomdp_maze):
        reunfolded = pomdp_maze.factory.set_imperfect_memory_size(2)
        assert reunfolded.feature_kind == "pomdp"
        assert reunfolded.parameter_space.num_parameters > pomdp_maze.colored_mdp.parameter_space.num_parameters

    def test_a_parameter_name_decodes_to_what_made_it(self, tmp_path):
        factory = observable_pomdp(tmp_path)
        factory.build()
        for obs in range(factory.observations):
            for memory in range(2):
                for is_action_parameter in (True, False):
                    name = factory.create_parameter_name(obs, memory, is_action_parameter)
                    assert factory.decode_parameter_name(name) == (is_action_parameter, obs, memory)

    @pytest.mark.xfail(strict=True, reason="Storm gives every observable the value of the first one, and prints a true boolean as nothing")
    def test_the_observations_have_labels_of_their_own(self, pomdp_maze):
        """The 8 observations of pomdp-maze are given by six boolean observables."""
        assert len(set(pomdp_maze.factory.observation_labels)) == 8

    def test_more_memory_for_one_observation_leaves_the_others_alone(self, pomdp_maze):
        factory = pomdp_maze.factory
        factory.set_imperfect_memory_size(1)
        obs = factory.observation_states.index(max(factory.observation_states))
        before = list(factory.observation_memory_size)
        factory.increase_memory_size(obs)
        assert factory.observation_memory_size == [size + (other == obs) for other, size in enumerate(before)]

    def test_memory_from_a_result_follows_each_kind_of_observation(self, pomdp_maze):
        """An observation of a single state gets 1 node, one with a size from the result at least one more than now, one with a cutoff that size, any other the
        limit."""
        factory = pomdp_maze.factory
        factory.set_imperfect_memory_size(1)
        shared = [obs for obs in range(factory.observations) if factory.observation_states[obs] > 1]
        single = [obs for obs in range(factory.observations) if factory.observation_states[obs] <= 1]
        assert len(shared) >= 3 and single
        factory.set_memory_from_result_new({shared[0]: 3, single[0]: 5}, {shared[1]: 2}, memory_limit=4)
        memory = factory.observation_memory_size
        assert memory[single[0]] == 1
        assert (memory[shared[0]], memory[shared[1]], memory[shared[2]]) == (3, 2, 4)
