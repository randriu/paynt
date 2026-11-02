import pytest
import stormpy

import paynt.dt
import paynt.dt.dtnest.task
import paynt.model.model_builder
import paynt.task
import paynt.utils.timer

from tests.helpers.models import get_sketch_paths


class TestDtSynthesis:
    def test_split_undecided_space_survives_a_multi_property_specification(self):
        """
        Regression test: scheduler_choices is only ever populated by ColoredMdp.scheduler_is_consistent's "dt" branch
        when the specification is single-property (see split_undecided_space/verify_parameter_space's guards) -- for any
        specification with more than one property (a constraint alongside the optimality objective, as
        constructed here), it stays None on every parameter space. split_undecided_space used to copy it unconditionally
        (parent_info.scheduler_choices = parameter_space.scheduler_choices) and verify_parameter_space used to
        iterate it unconditionally too, so the very first split of a multi-property parameter space raised
        AttributeError -- silently never caught because every DT sketch actually used elsewhere in this
        suite happens to be single-property.
        """
        sketch_path, _ = get_sketch_paths("tests/dt-orchard")
        prism = stormpy.parse_prism_program(sketch_path, prism_compatibility=True)
        # a trivially-true constraint alongside the real optimality objective: enough to make
        # is_single_property False without changing what the "right" answer is
        properties = stormpy.parse_properties('P>=0 [F "goal"]; Pmax=? [F "goal"]', prism)
        assert len(properties) == 2

        explicit_model = paynt.model.model_builder.ModelBuilder.from_prism(prism, None, False)
        task = paynt.task.SynthesisTask(properties, timeout=15)
        build_task = paynt.dt.dtnest.task.DtNestTask(error_threshold=0.05, tree_depth=2)
        assert not task.specification.is_single_property

        factory = paynt.dt.DtColoredMdpFactory(explicit_model, build_task)
        synthesizer = paynt.dt.DtSynthesizer(factory, task)
        synthesizer.synthesize_tree(depth=2, timeout=15)
        assert synthesizer.best_tree is not None

    def test_split_undecided_space_survives_a_genuinely_multi_constraint_specification(self):
        """
        Regression coverage for the multi-constraint AR splitting gap (paper https://www.jair.org/index.php/jair/article/view/16593 Section 3.3): unlike
        test_split_undecided_space_survives_a_multi_property_specification above (one trivially-true
        constraint alongside optimality, which resolves before ever reaching split_undecided_space's new
        L(h)/undecided_results() logic in any interesting way), this specification has TWO real constraints
        and NO optimality objective, forcing split_undecided_space to actually run its new
        compute_incompatibility_levels/undecided_results() code paths repeatedly.

        dt-orchard's only event is "goal", so two P>=x[F "goal"] constraints at different thresholds don't
        produce genuine cross-constraint disagreement (both push toward the same underlying policy) -- this
        model can't exercise the L(h) tier specifically (that needs constraints on genuinely different
        criteria, like easy.props's two different reward structures in
        tests/cli/test_multi_constraint_splitting.py). What it does confirm: neither harmonize_inconsistent_
        scheduler nor split_undecided_space crashes or hangs when node.analysis_result.constraints_result has
        2 undecided entries, and the search genuinely explores (unlike a trivially-resolved-at-the-root spec).
        Bounded rather than run to completion -- this threshold has no admissible tree within a few seconds
        on this model, confirmed manually, and this test only cares about "does not crash", not convergence.
        """
        sketch_path, _ = get_sketch_paths("tests/dt-orchard")
        prism = stormpy.parse_prism_program(sketch_path, prism_compatibility=True)
        properties = stormpy.parse_properties('P>=0.55 [F "goal"]; P>=0.5 [F "goal"]', prism)
        assert len(properties) == 2

        explicit_model = paynt.model.model_builder.ModelBuilder.from_prism(prism, None, False)
        task = paynt.task.SynthesisTask(properties, timeout=8)
        build_task = paynt.dt.dtnest.task.DtNestTask(error_threshold=0.05, tree_depth=2)
        assert len(task.specification.constraints) == 2
        assert task.specification.optimality is None

        factory = paynt.dt.DtColoredMdpFactory(explicit_model, build_task)
        synthesizer = paynt.dt.DtSynthesizer(factory, task)
        paynt.utils.timer.GlobalTimer.start(8)
        try:
            synthesizer.synthesize_tree(depth=2, timeout=8)
        finally:
            paynt.utils.timer.GlobalTimer.start()

    def test_synthesize_finds_the_known_optimum(self, dt_orchard):
        """Regression test for DtSynthesizer.split_undecided_space/scheduler_scores (the search-policy methods moved off the factory during this migration).
        Kept at depth 0 deliberately: full AR-based tree synthesis is combinatorially expensive on this model at higher depths (which is exactly why
        get_dt_with_api.py uses dtnest's subtree-based search instead of plain synthesize_tree for real use) -- but dtnest's subtree calls go through this same
        split_undecided_space override internally, and that path is already verified end-to-end (byte-identical value 0.6313574509764854) via
        get_dt_with_api.py.

        Since synthesis is deterministic, the tree itself (not just its value) is checked too: at depth 0 the only admissible tree is a single leaf, so its
        string form is a one-line action label.
        """
        synthesizer = paynt.dt.DtSynthesizer(dt_orchard.factory, dt_orchard.task)
        synthesizer.synthesize_tree(depth=0)
        assert synthesizer.best_tree is not None
        assert synthesizer.best_tree_value == pytest.approx(0.48450450140758644, abs=1e-6)
        assert synthesizer.best_tree.to_string() == "roll\n"
