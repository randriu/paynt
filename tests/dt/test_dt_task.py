import pytest
import stormpy

import paynt.task
import paynt.dt.task
import paynt.dt.dtnest.task


def _reward_properties():
    return stormpy.parse_properties_without_context('R{"steps"}min=? [F "goal"]')


class TestDtTask:
    """DtTask carries only DT's build-time settings (tree_depth, ...) -- no specification/timeout/ use_exact, which now live on the separate, non-inheriting
    paynt.task.SynthesisTask instead (see TestDtNestTask/test_task.py for that side)."""

    def test_tree_depth_is_stored(self):
        build_task = paynt.dt.task.DtTask(tree_depth=3)
        assert build_task.tree_depth == 3

    def test_scheduler_to_map_starts_unset(self):
        build_task = paynt.dt.task.DtTask(tree_depth=3)
        assert not build_task.has_scheduler_to_map
        build_task.set_scheduler_to_map("some-scheduler")
        assert build_task.has_scheduler_to_map

    def test_use_exact_is_no_longer_hardcoded_to_false(self):
        """Regression test: the pre-split DtTask.__init__ used to hardcode Property.initialize(False) regardless of the caller's use_exact, silently ignoring
        it.

        Since DtTask no longer touches specification construction at all (that's now exclusively SynthesisTask's job -- see paynt.task.SynthesisTask.__init__),
        there is no longer a DT-specific code path where this bug could be reintroduced; kept here as a direct pin on the pairing DT actually uses.
        """
        with pytest.raises(ValueError):
            paynt.task.SynthesisTask(_reward_properties(), use_exact=True)
        # sanity: the same construction succeeds without use_exact
        paynt.task.SynthesisTask(_reward_properties(), use_exact=False)


class TestDtNestTask:
    def test_is_a_dt_build_task(self):
        build_task = paynt.dt.dtnest.task.DtNestTask(error_threshold=0.05, tree_depth=5)
        assert isinstance(build_task, paynt.dt.task.DtTask)
        assert build_task.error_threshold == 0.05
        assert build_task.tree_depth == 5

    def test_initial_tree_defaults_to_none(self):
        build_task = paynt.dt.dtnest.task.DtNestTask(error_threshold=0.05)
        assert build_task.initial_tree is None

    def test_the_depth_of_the_tree_is_that_of_dt_task(self):
        """DtNestTask does not touch the depth of the tree to synthesize: its default is that of DtTask, and it is not the depth of the subtrees."""
        build_task = paynt.dt.dtnest.task.DtNestTask()
        assert build_task.tree_depth == paynt.dt.task.DtTask().tree_depth == 0

    def test_the_depth_of_the_subtrees_is_a_setting_of_its_own(self):
        assert paynt.dt.dtnest.task.DtNestTask().max_subtree_depth == 7
        build_task = paynt.dt.dtnest.task.DtNestTask(tree_depth=2, max_subtree_depth=4)
        assert (build_task.tree_depth, build_task.max_subtree_depth) == (2, 4)
        build_task = paynt.dt.dtnest.task.DtNestTask(max_subtree_depth=3)
        assert (build_task.tree_depth, build_task.max_subtree_depth) == (0, 3)
