import pytest

import paynt.colored_mdp
import paynt.dt
import paynt.dt._utils
import paynt.synthesizer.search_node


class TestDtColoredMdpFactory:
    def test_load_sketch_produces_a_dt_colored_mdp(self, dt_colored_mdp):
        """An MDP-with-parameters sketch that doesn't have PRISM-declared parameters (no jani_unfolder) and isn't partially observable resolves through
        DtColoredMdpFactory."""
        assert type(dt_colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert dt_colored_mdp.feature_kind == "dt"
        assert isinstance(dt_colored_mdp.feature_info, paynt.dt._utils.DtInfo)

    def test_factory_does_not_build_automatically(self, dt_colored_mdp_factory):
        """Guardrail for the "factories don't eagerly build" redesign: a freshly-constructed factory holds no colored_mdp/memory-size state of its own --
        reset_tree()/build() must be called explicitly to get one, matching every other specialist factory."""
        assert isinstance(dt_colored_mdp_factory, paynt.dt.DtColoredMdpFactory)
        assert not hasattr(dt_colored_mdp_factory, "colored_mdp")

    @pytest.mark.parametrize("depth", [0, 2])
    def test_build_produces_a_tree_at_build_tasks_configured_depth(self, dt_colored_mdp_factory, depth):
        """Build() must honor build_task.tree_depth, not silently default to 0 (dt-orchard's fixture goes through DtNestTask, since Sketch.load_sketch doesn't
        yet know whether the caller wants dtnest)."""
        assert dt_colored_mdp_factory.build_task is not None
        dt_colored_mdp_factory.build_task.tree_depth = depth
        colored_mdp = dt_colored_mdp_factory.build()
        assert colored_mdp.feature_info.decision_tree.get_depth() == depth

    def test_a_sketch_loaded_without_settings_builds_a_tree_of_depth_0(self, dt_colored_mdp_factory):
        """The default of the library is that of the command line (--tree-depth 0), though the task that Sketch.load_sketch makes is the one of dtnest."""
        assert dt_colored_mdp_factory.build_task is not None
        assert dt_colored_mdp_factory.build_task.tree_depth == 0
        assert dt_colored_mdp_factory.build().feature_info.decision_tree.get_depth() == 0

    def test_reset_tree_produces_a_fresh_colored_mdp(self, dt_colored_mdp, dt_colored_mdp_factory):
        reset = dt_colored_mdp_factory.reset_tree(2)
        assert reset.feature_kind == "dt"
        assert reset.feature_info.decision_tree.get_depth() == 2
        # the identity data (computed once at construction) must carry over unchanged across reset_tree
        assert reset.feature_info.action_labels is dt_colored_mdp.feature_info.action_labels

    def test_build_produces_an_mdp(self, dt_colored_mdp):
        node = paynt.synthesizer.search_node.SearchNode(dt_colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = dt_colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_factory_without_a_task_matches_the_get_dt_with_api_pattern(self, dt_colored_mdp):
        factory = paynt.dt.DtColoredMdpFactory(dt_colored_mdp.underlying_mdp)
        assert factory.build_task is None
        assert factory.reset_tree(0).feature_info.decision_tree.get_depth() == 0
