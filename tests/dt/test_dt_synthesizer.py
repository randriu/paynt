import pytest

import paynt.dt

from tests.helpers.models import load_sketch


class TestDtSynthesis:
    def test_synthesize_finds_the_known_optimum(self, dt_orchard):
        """At depth 0 a tree is a single leaf: the best one is `roll`."""
        synthesizer = paynt.dt.DtSynthesizer(dt_orchard.factory, dt_orchard.task)
        synthesizer.synthesize_tree(depth=0)
        assert synthesizer.best_tree is not None
        # SMPMC finds the same, see test_smpmc_inner_engine.py
        assert synthesizer.best_tree_value == pytest.approx(0.48450450140758644, abs=1e-6)
        assert synthesizer.best_tree.to_string() == "roll\n"


class TestTwoConstraints:
    """The trips of models/tests/time-energy as trees, with tree.props: the two stages need different speeds."""

    def synthesize(self, depth):
        factory, task = load_sketch("tests/time-energy", sketch_name="tree.templ", props_name="tree.props")
        synthesizer = paynt.dt.DtSynthesizer(factory, task)
        synthesizer.synthesize_tree(depth)
        return synthesizer.best_tree

    def test_no_tree_of_depth_0_satisfies_both(self):
        assert self.synthesize(0) is None

    def test_a_tree_of_depth_1_drives_one_stage_fast(self):
        tree = self.synthesize(1)
        leaves = sorted(tree.action_labels[leaf.action] for leaf in tree.collect_terminals())
        assert leaves in (["fast", "medium"], ["__random__", "fast"])
