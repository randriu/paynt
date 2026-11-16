import pytest
import stormpy

import paynt.specification.property


def _reachability_properties():
    return stormpy.parse_properties('Pmax=? [F "goal"]')


class TestConstructSpecification:
    def test_builds_specification_from_raw_properties(self):
        spec = paynt.specification.property.construct_specification(_reachability_properties())
        assert spec.num_properties == 1
        assert spec.has_optimality


class TestSoundMode:
    """--sound: Property.initialize must not cap minmax iterations, since a capped solve that doesn't converge
    returns an unconverged value that pruning would then trust."""

    def _minmax_iterations(self, monkeypatch, sound):
        monkeypatch.setattr(paynt.specification.property.Property, "sound", sound)
        paynt.specification.property.Property.initialize()
        return paynt.specification.property.Property.environment.solver().min_max().maximal_number_of_iterations

    def test_default_caps_minmax_iterations(self, monkeypatch):
        assert self._minmax_iterations(monkeypatch, sound=False) == paynt.specification.property.Property.max_minmax_iterations

    def test_sound_applies_no_cap(self, monkeypatch):
        uncapped = stormpy.Environment().solver().min_max().maximal_number_of_iterations
        assert self._minmax_iterations(monkeypatch, sound=True) == uncapped

    def test_load_sketch_sets_sound(self, monkeypatch):
        import paynt.parser.sketch
        from tests.helpers.models import get_sketch_paths

        monkeypatch.setattr(paynt.specification.property.Property, "sound", False)
        paynt.parser.sketch.Sketch.load_sketch(*get_sketch_paths("tests/smpmc-tiny"), sound=True)
        assert paynt.specification.property.Property.sound is True


class TestSpecificationRewrap:
    def test_preserves_property_type_and_epsilon(self):
        """JaniUnfolder needs this: after translating PRISM to JANI, property atoms may change, so the specification's properties must be rebuilt from new raw
        stormpy formulas while preserving each property's paynt-level type (Property vs OptimalityProperty) and epsilon."""
        original = paynt.specification.property.construct_specification(_reachability_properties(), relative_error=0.1)
        assert original.has_optimality
        original_epsilon = original.optimality.epsilon
        assert original_epsilon == 0.1

        # simulate what JaniUnfolder does: re-wrap around "new" (here: the same) raw stormpy formulas
        new_raw_properties = [p.property for p in original.all_properties()]
        rewrapped = original.rewrap(new_raw_properties)

        assert type(rewrapped.optimality) is type(original.optimality)
        assert rewrapped.optimality.epsilon == original_epsilon
        assert rewrapped.num_properties == original.num_properties


class TestThreshold:
    # Storm alone makes 0.2 the double 0.19999999999999998, so a design whose probability is 0.2 would violate P<=0.2
    @pytest.mark.parametrize("threshold", ["0.25", "0.3", "0.2", "0.1", "-0.1", "1e-3"])
    def test_is_the_double_nearest_to_the_written_number(self, threshold):
        prop = paynt.specification.property.Property(stormpy.parse_properties(f'P<={threshold} [F "goal"]')[0])
        assert prop.threshold == float(threshold)


WALK = """dtmc

module m
    s : [0..2] init 0;
    [go] s<2 -> (s'=s+1);
    [] s=2 -> true;
endmodule

label "goal" = s=2;

rewards "cost"
    [go] true : 1;
endrewards
"""


class TestHelpers:
    def test_construct_reward_property_optimizes_the_reward_to_the_target(self):
        prop = paynt.specification.property.construct_reward_property("cost", minimizing=True, target_label="goal")
        assert type(prop) is paynt.specification.property.OptimalityProperty
        assert prop.minimizing
        assert str(prop.formula) == 'R{"cost"}min=? [F "goal"]'

    @pytest.mark.parametrize(
        ("text", "optimality"),
        [('P>=0.5 [F "goal"]', 'Pmax=? [F "goal"]'), ('R{"cost"}<=3 [F "goal"]', 'R{"cost"}min=? [F "goal"]'), ("P<0.5 [F s=2]", "Pmin=? [F (s = 2)]")],
    )
    def test_a_threshold_becomes_the_optimum_in_its_direction(self, tmp_path, text, optimality):
        (tmp_path / "walk.prism").write_text(WALK)
        program = stormpy.parse_prism_program(str(tmp_path / "walk.prism"))
        prop = paynt.specification.property.Property(stormpy.parse_properties(text, program)[0])
        assert str(prop.transform_to_optimality_formula(program).get_raw_formula()) == optimality

    @pytest.mark.parametrize("direction", ["max", "min"])
    def test_a_suboptimal_value_is_just_worse_than_the_optimum(self, direction):
        prop = paynt.specification.property.OptimalityProperty(stormpy.parse_properties(f'P{direction}=? [F "goal"]')[0])
        prop.update_optimum(0.5)
        worse = prop.suboptimal_value()
        assert worse == pytest.approx(0.5, abs=paynt.specification.property.Property.model_checking_precision)
        assert worse < 0.5 if direction == "max" else worse > 0.5

    def test_only_a_game_property_has_a_game_formula(self, posmg_test_game, smpmc_tiny):
        assert posmg_test_game.task.get_property().has_game_formula
        assert not smpmc_tiny.task.get_property().has_game_formula


class TestUntil:
    """models/tests/edge-cases/until.templ: H=0 reaches the goal only through the bad state, so `!"bad" U "goal"` and `F "goal"` differ there."""

    @staticmethod
    def specification(tmp_path, *properties):
        import paynt.parser.sketch
        from tests.helpers.models import get_sketch_paths

        props_path = tmp_path / "case.props"
        props_path.write_text("\n".join(properties) + "\n")
        _factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths("tests/edge-cases", "until.templ")[0], str(props_path))
        return task.specification

    def test_is_checked_as_reaching_the_goal_when_storm_makes_the_bad_state_absorbing(self, tmp_path):
        """Storm does for a model built for this formula alone, with atomic sides."""
        [prop] = self.specification(tmp_path, 'Pmax=? ["ok" U "goal"]').all_properties()
        assert str(prop.formula) == 'Pmax=? [F "goal"]'

    @pytest.mark.parametrize(
        "properties", [['Pmax=? [!"bad" U "goal"]'], ['P>=0.1 [F "goal"]', 'Pmax=? ["ok" U "goal"]']], ids=["a negated side", "with another property"]
    )
    def test_stays_when_the_bad_state_can_reach_the_goal(self, tmp_path, properties):
        prop = self.specification(tmp_path, *properties).all_properties()[-1]
        assert prop.is_until

    def test_what_needs_a_target_refuses_it(self, tmp_path):
        prop = self.specification(tmp_path, 'Pmax=? [!"bad" U "goal"]').all_properties()[0]
        with pytest.raises(NotImplementedError, match="is checked as an until formula"):
            prop.get_target_label()


@pytest.mark.parametrize(
    "text, bounded",
    [
        ('P>=0.5 [F<=2 "goal"]', True),
        ('P>=0.5 ["ok" U<=2 "goal"]', True),
        ('P>=0.5 [F[1,2] "goal"]', True),
        ('P>=0.5 [X "goal"]', True),
        ('R{"steps"}<=1 [C<=2]', True),
        ('R{"steps"}<=1 [I=2]', True),
        ('P>=0.5 [F "goal"]', False),
        ('P>=0.5 ["ok" U "goal"]', False),
        ('R{"steps"}<=1 [F "goal"]', False),
        ('R{"steps"}<=1 [C]', False),
    ],
)
def test_a_property_is_step_bounded(text, bounded):
    prop = paynt.specification.property.Property(stormpy.parse_properties(text)[0])
    assert prop.is_step_bounded == bounded
