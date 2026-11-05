from fractions import Fraction

import pytest

import paynt.colored_mdp
import paynt.dt
import paynt.mdp_family
import paynt.mdp_family.pomdp
import paynt.parser.sketch as sketch
import paynt.pomdp
import paynt.pomdp.decpomdp
import paynt.pomdp.posmg

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

    @pytest.mark.parametrize(
        ("project", "factory_class"),
        [
            ("dt_orchard", paynt.dt.DtColoredMdpFactory),
            ("mdp_family_avoid_8_2_easy", paynt.mdp_family.MdpFamilyColoredMdpFactory),
            ("pomdp_family_avoid_smaller", paynt.mdp_family.pomdp.PomdpFamilyColoredMdpFactory),
            ("pomdp_maze", paynt.pomdp.PomdpColoredMdpFactory),
            ("decpomdp_dectiger", paynt.pomdp.decpomdp.DecPomdpColoredMdpFactory),
            ("posmg_mec_test", paynt.pomdp.posmg.PosmgColoredMdpFactory),
        ],
    )
    def test_any_other_sketch_gives_a_factory_that_builds_only_when_asked(self, request, project, factory_class):
        factory = request.getfixturevalue(project).factory
        assert type(factory) is factory_class
        assert not hasattr(factory, "colored_mdp")
        assert not hasattr(factory, "current_memory_size")


RANGES = """dtmc

hole int A in {0..6:2};
hole double P in {0.1..0.5:0.2};
hole int B in {1..3};

module m
    s : [0..2] init 0;
    [go] s=0 -> P : (s'=1) + 1-P : (s'=2);
    [] s>0 -> true;
endmodule

label "goal" = s=1;

rewards "cost"
    [go] true : A+B;
endrewards
"""

OBSERVABLES = """pomdp

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


def load_text(tmp_path, text):
    (tmp_path / "sketch.templ").write_text(text)
    (tmp_path / "sketch.props").write_text('P>=0.2 [F "goal"]\n')
    return sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))


class TestHoleOptions:
    def test_ranges_with_a_step_and_decimal_ranges_give_every_option(self, tmp_path):
        space = load_text(tmp_path, RANGES)[0].build().parameter_space
        options = {space.parameter_name(parameter): [Fraction(label) for label in space.parameter_to_option_labels[parameter]] for parameter in range(3)}
        assert options == {"A": [0, 2, 4, 6], "P": [Fraction(1, 10), Fraction(3, 10), Fraction(1, 2)], "B": [1, 2, 3]}


def test_an_observables_block_makes_the_observations(tmp_path):
    """The observable o is 0 in the initial state and 1 everywhere else."""
    pomdp = load_text(tmp_path, OBSERVABLES)[0].pomdp
    assert pomdp.nr_observations == 2
    assert pomdp.observations[1] == pomdp.observations[2] != pomdp.observations[0]


def test_a_missing_sketch_is_reported(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        sketch.Sketch.load_sketch(str(tmp_path / "missing.templ"), str(tmp_path / "missing.props"))
