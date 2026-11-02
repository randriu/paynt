import paynt.parser.sketch as sketch
import paynt.colored_mdp

from tests.helpers.models import get_sketch_paths


class TestSketch:
    def test_load_sketch_mdp_prism(self):
        # setup
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")

        # test
        colored_mdp_factory, task = sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = colored_mdp_factory.colored_mdp

        # assert
        assert isinstance(colored_mdp_factory, paynt.colored_mdp.IdentityColoredMdpFactory)
        assert type(colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert colored_mdp.feature_kind == "generic"
        assert colored_mdp.underlying_mdp.nr_states == 183
        # TODO add more asserts

    # add more tests
