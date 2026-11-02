import pytest

import paynt.colored_mdp
import paynt.dt
import paynt.dt._utils
import paynt.synthesizer.search_node


class TestDtColoredMdpFactory:
    def test_load_sketch_produces_a_dt_colored_mdp(self, dt_orchard):
        """An MDP-with-parameters sketch that doesn't have PRISM-declared parameters (no jani_unfolder) and isn't partially observable resolves through
        DtColoredMdpFactory."""
        assert type(dt_orchard.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert dt_orchard.colored_mdp.feature_kind == "dt"
        assert isinstance(dt_orchard.colored_mdp.feature_info, paynt.dt._utils.DtInfo)

    def test_factory_does_not_build_automatically(self, dt_orchard):
        """Guardrail for the "factories don't eagerly build" redesign: a freshly-constructed factory holds no colored_mdp/memory-size state of its own --
        reset_tree()/build() must be called explicitly to get one, matching every other specialist factory."""
        assert isinstance(dt_orchard.factory, paynt.dt.DtColoredMdpFactory)
        assert not hasattr(dt_orchard.factory, "colored_mdp")

    @pytest.mark.parametrize("depth", [0, 2])
    def test_build_produces_a_tree_at_build_tasks_configured_depth(self, dt_orchard, depth):
        """The factory's build() honors build_task.tree_depth (the dt-orchard fixture goes through DtNestTask, since Sketch.load_sketch does not know whether
        the caller wants dtnest)."""
        assert dt_orchard.factory.build_task is not None
        dt_orchard.factory.build_task.tree_depth = depth
        colored_mdp = dt_orchard.factory.build()
        assert colored_mdp.feature_info.decision_tree.get_depth() == depth

    def test_a_sketch_loaded_without_settings_builds_a_tree_of_depth_0(self, dt_orchard):
        """The default of the library is that of the command line (--tree-depth 0), though the task that Sketch.load_sketch makes is the one of dtnest."""
        assert dt_orchard.factory.build_task is not None
        assert dt_orchard.factory.build_task.tree_depth == 0
        assert dt_orchard.factory.build().feature_info.decision_tree.get_depth() == 0

    def test_reset_tree_produces_a_fresh_colored_mdp(self, dt_orchard):
        reset = dt_orchard.factory.reset_tree(2)
        assert reset.feature_kind == "dt"
        assert reset.feature_info.decision_tree.get_depth() == 2
        # the identity data (computed once at construction) must carry over unchanged across reset_tree
        assert reset.feature_info.action_labels is dt_orchard.colored_mdp.feature_info.action_labels

    def test_build_produces_an_mdp(self, dt_orchard):
        node = paynt.synthesizer.search_node.SearchNode(dt_orchard.colored_mdp.parameter_space.copy())
        node.mdp, node.selected_choices = dt_orchard.colored_mdp.build(node.parameter_space)
        assert node.mdp.states > 0

    def test_factory_without_a_task_matches_the_get_dt_with_api_pattern(self, dt_orchard):
        factory = paynt.dt.DtColoredMdpFactory(dt_orchard.colored_mdp.underlying_mdp)
        assert factory.build_task is None
        assert factory.reset_tree(0).feature_info.decision_tree.get_depth() == 0
