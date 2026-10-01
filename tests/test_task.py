import pytest
import stormpy

import paynt.task


def _reachability_properties():
    return stormpy.parse_properties_without_context('Pmax=? [F "goal"]')


def _reward_properties():
    return stormpy.parse_properties_without_context('R{"steps"}min=? [F "goal"]')


class TestSynthesisTask:
    def test_specification_is_constructed_from_properties(self):
        task = paynt.task.SynthesisTask(_reachability_properties())
        assert task.specification.num_properties == 1
        assert task.specification.has_optimality

    def test_get_property_matches_specification(self):
        task = paynt.task.SynthesisTask(_reachability_properties())
        assert task.get_property() is task.specification.all_properties()[0]

    def test_timeout_and_use_exact_are_stored(self):
        task = paynt.task.SynthesisTask(_reachability_properties(), timeout=42, use_exact=False)
        assert task.timeout == 42
        assert task.use_exact is False

    def test_from_specification_wraps_without_reparsing(self):
        original = paynt.task.SynthesisTask(_reachability_properties(), timeout=10, use_exact=False)
        wrapped = paynt.task.SynthesisTask.from_specification(original.specification, timeout=99, use_exact=False)
        assert wrapped.specification is original.specification
        assert wrapped.timeout == 99

    def test_from_specification_keeps_the_robust_synthesis_fields(self):
        original = paynt.task.SynthesisTask(_reachability_properties())
        wrapped = paynt.task.SynthesisTask.from_specification(original.specification, forall_pattern="^o", verify_robust=True)
        assert wrapped.forall_pattern == "^o"
        assert wrapped.verify_robust is True

    def test_use_exact_reaches_construct_property(self):
        """use_exact must actually flow through SynthesisTask -> construct_specification -> construct_property, which rejects reward properties when
        use_exact=True."""
        with pytest.raises(ValueError):
            paynt.task.SynthesisTask(_reward_properties(), use_exact=True)
        # sanity: the same property without use_exact does not raise
        paynt.task.SynthesisTask(_reward_properties(), use_exact=False)
