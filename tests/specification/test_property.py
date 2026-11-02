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

        # simulate what JaniUnfolder does: re-wrap around "new" (here: the same) raw stormpy formulas
        new_raw_properties = [p.property for p in original.all_properties()]
        rewrapped = original.rewrap(new_raw_properties)

        assert type(rewrapped.optimality) is type(original.optimality)
        assert rewrapped.optimality.epsilon == original_epsilon
        assert rewrapped.num_properties == original.num_properties
