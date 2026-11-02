import paynt.parser.sketch as sketch
import paynt.colored_mdp

from tests.helpers.models import get_sketch_paths


class TestSketch:
    def test_a_dtmc_sketch_gives_a_generic_colored_mdp(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        colored_mdp_factory, task = sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = colored_mdp_factory.colored_mdp
        assert isinstance(colored_mdp_factory, paynt.colored_mdp.IdentityColoredMdpFactory)
        assert type(colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert colored_mdp.feature_kind == "generic"
        assert colored_mdp.underlying_mdp.nr_states == 183
