import paynt.parameter_space.parameter_space
import paynt.synthesizer.search_node


def _parameter_space_with_two_binary_parameters():
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    parameter_space.add_parameter("x", ["a", "b"])
    parameter_space.add_parameter("y", ["c", "d"])
    return parameter_space


class _FakeConstraintsResult:
    undecided_constraints = []


class _FakeAnalysisResult:
    constraints_result = _FakeConstraintsResult()


class TestSearchNode:
    def test_split_produces_children_with_incremented_refinement_depth_and_shared_parent_info(self):
        """Every child of a split gets the same ParentInfo, a snapshot of the parent."""
        parent = paynt.synthesizer.search_node.SearchNode(_parameter_space_with_two_binary_parameters())
        parent.selected_choices = [True, False]
        parent.analysis_result = _FakeAnalysisResult()
        children = parent.split(0, [[0], [1]])
        assert len(children) == 2
        for child in children:
            assert child.refinement_depth == 1
            # constraint_indices comes from the parent's analysis result (which constraints are still
            # undecided), not copied unchanged from the parent node's own constraint_indices
            assert child.constraint_indices == _FakeConstraintsResult.undecided_constraints
            assert child.parent_info is not None
            assert child.parent_info.selected_choices == [True, False]
        # every child shares the identical ParentInfo object, not independent copies
        assert children[0].parent_info is children[1].parent_info
        # the split itself only narrows the value (native), not the node's own search state
        assert children[0].parameter_space.parameter_options(0) == [0]
        assert children[1].parameter_space.parameter_options(0) == [1]

    def test_split_produces_children_of_the_same_concrete_type(self):
        """DtSearchNode/PolicyTreeNode rely on split() constructing children via type(self), not a hardcoded SearchNode, so a subclass's own extra fields (e.g.
        scheduler_choices) get properly initialized on every child without split() needing to know about them."""

        class FakeSearchNode(paynt.synthesizer.search_node.SearchNode):
            def __init__(self, parameter_space, parent_info=None):
                super().__init__(parameter_space, parent_info)
                self.extra = "default"

        parent = FakeSearchNode(_parameter_space_with_two_binary_parameters())
        parent.analysis_result = _FakeAnalysisResult()
        children = parent.split(0, [[0], [1]])
        assert all(isinstance(child, FakeSearchNode) for child in children)
        assert all(child.extra == "default" for child in children)

    def test_collect_parent_info_snapshots_undecided_constraints(self):
        analysis_result = _FakeAnalysisResult()
        analysis_result.constraints_result = _FakeConstraintsResult()
        analysis_result.constraints_result.undecided_constraints = [1, 2]

        node = paynt.synthesizer.search_node.SearchNode(_parameter_space_with_two_binary_parameters())
        node.selected_choices = [True]
        node.refinement_depth = 3
        node.analysis_result = analysis_result

        parent_info = node.collect_parent_info()
        assert parent_info.selected_choices == [True]
        assert parent_info.refinement_depth == 3
        assert parent_info.constraint_indices == [1, 2]
        # DT-only fields default to None for every other caller
        assert parent_info.analysis_result is None
        assert parent_info.scheduler_choices is None
