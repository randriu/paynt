import pytest
import stormpy

import paynt.dt
import paynt.dt.decision_tree
import paynt.dt.dtnest
import paynt.dt.dtnest.api
import paynt.model.model_builder
import paynt.task

from tests.helpers.models import get_sketch_paths, load_sketch

# the value of the don't-care action everywhere (solver output, a regression value); the best scheduler of the MDP gives 0.6314
RANDOM = 0.48451914218945663


def _dtnest_task_and_factory(properties_string, error_threshold=0.05, timeout=30):
    """The task, the dtnest settings and the factory of dt-orchard with the given properties."""
    sketch_path, _ = get_sketch_paths("tests/dt-orchard")
    prism = stormpy.parse_prism_program(sketch_path, prism_compatibility=True)
    properties = stormpy.parse_properties(properties_string, prism)
    explicit_model = paynt.model.model_builder.ModelBuilder.from_prism(prism, None, False)
    task = paynt.task.SynthesisTask(properties, timeout=timeout)
    build_task = paynt.dt.dtnest.DtNestTask(error_threshold=error_threshold)
    factory = paynt.dt.DtColoredMdpFactory(explicit_model, build_task)
    return task, build_task, factory


class TestDtNestConstraintHandling:
    """A single constraint becomes an objective for dtnest, with its threshold as the target (reduce_constraint_to_optimality)."""

    def test_constraint_between_random_and_optimal_is_treated_as_its_own_threshold(self):
        """0.55 is between the value of the don't-care action and the optimum."""
        task, build_task, factory = _dtnest_task_and_factory('P>=0.55 [F "goal"]')
        # subtrees of depth 1 keep the search short
        build_task.max_subtree_depth = 1
        result = paynt.dt.dtnest.synthesize(factory, task, build_task)
        assert result.success
        assert result.value >= 0.55

    def test_constraint_already_satisfied_by_random_returns_it_directly(self):
        """The don't-care action already gives more than 0.3, so no subtree is searched."""
        task, build_task, factory = _dtnest_task_and_factory('P>=0.3 [F "goal"]')
        result = paynt.dt.dtnest.synthesize(factory, task, build_task)
        assert result.success
        assert result.value == pytest.approx(RANDOM, abs=1e-6)

    def test_constraint_above_optimal_is_unsatisfiable(self):
        """0.9 is above the optimum of the MDP."""
        task, build_task, factory = _dtnest_task_and_factory('P>=0.9 [F "goal"]')
        result = paynt.dt.dtnest.synthesize(factory, task, build_task)
        assert not result.success

    def test_optimality_alongside_a_constraint_is_rejected(self):
        """A constraint next to an objective is more than dtnest can enforce."""
        task, build_task, factory = _dtnest_task_and_factory('Pmax=? [F "goal"]; P>=0.5 [F "goal"]')
        with pytest.raises(ValueError):
            paynt.dt.dtnest.synthesize(factory, task, build_task)

    def test_multiple_constraints_without_optimality_is_rejected(self):
        """Two constraints do not make one objective."""
        task, build_task, factory = _dtnest_task_and_factory('P>=0.5 [F "goal"]; P<=0.9 [F "goal"]')
        with pytest.raises(ValueError):
            paynt.dt.dtnest.synthesize(factory, task, build_task)


def _node(identifier, old_identifier):
    node = paynt.dt.decision_tree.DecisionTreeNode(None)
    node.identifier = identifier
    node.old_identifier = old_identifier
    return node


class TestRemapNodeQueueAfterReplacement:
    """After a subtree is replaced, the queued nodes are renamed to their identifiers in the new tree (old_identifier).

    A queued node inside the replaced subtree has no counterpart any more and leaves the queue.
    """

    def test_matching_entry_is_remapped_to_the_new_identifier(self):
        tree = paynt.dt.decision_tree.DecisionTree([], [])
        tree.root = _node(identifier=10, old_identifier=3)
        node_queue = [{"id": 3, "extra": "kept"}]
        result = paynt.dt.dtnest.DtNest.remap_node_queue_after_replacement(node_queue, tree)
        assert len(result) == 1
        assert result[0]["id"] == 10
        assert result[0]["extra"] == "kept"

    def test_entry_with_no_surviving_counterpart_is_dropped_not_raised(self):
        tree = paynt.dt.decision_tree.DecisionTree([], [])
        tree.root = _node(identifier=0, old_identifier=0)
        node_queue = [{"id": 0}, {"id": 999}]
        result = paynt.dt.dtnest.DtNest.remap_node_queue_after_replacement(node_queue, tree)
        assert len(result) == 1
        assert result[0]["id"] == 0

    def test_multiple_counterparts_raises(self):
        """assign_identifiers(keep_old=True) gives unique old identifiers, two matches are a bug."""
        root = _node(identifier=0, old_identifier=5)
        root.child_true = _node(identifier=1, old_identifier=5)
        root.child_false = _node(identifier=2, old_identifier=2)
        tree = paynt.dt.decision_tree.DecisionTree([], [])
        tree.root = root
        node_queue = [{"id": 5}]
        with pytest.raises(AssertionError):
            paynt.dt.dtnest.DtNest.remap_node_queue_after_replacement(node_queue, tree)


class TestDtNestDepths:
    """The depth of the tree (DtTask.tree_depth) and the depth of the subtrees of the dtnest iteration (DtNestTask.max_subtree_depth) are two settings: the
    second stays inside the iteration, where it limits the depth of the subtrees that are synthesized and is the depth of the subtrees put into the queue for
    replacement."""

    def test_the_subtree_depth_of_the_iteration_is_max_subtree_depth_whatever_the_tree_depth(self):
        _task, build_task, factory = _dtnest_task_and_factory('P>=0.55 [F "goal"]')
        build_task.tree_depth = 1
        build_task.max_subtree_depth = 4
        synthesizer = paynt.dt.dtnest.DtNest(factory, _task)
        assert synthesizer.subtree_depth == 4
        # the tree that the synthesizer starts from is that of the tree depth, which dtnest does not change
        assert synthesizer.colored_mdp.feature_info.decision_tree.get_depth() == 1
        assert build_task.tree_depth == 1

    def test_the_default_subtree_depth_does_not_depend_on_the_tree_depth(self):
        _task, build_task, factory = _dtnest_task_and_factory('P>=0.55 [F "goal"]')
        assert (build_task.tree_depth, build_task.max_subtree_depth) == (0, 7)
        assert paynt.dt.dtnest.DtNest(factory, _task).subtree_depth == 7

    def test_the_api_hands_the_subtree_depth_to_the_iteration(self, monkeypatch):
        """Not the depth of the tree: with a tree depth of 2 and subtrees of depth 5, the iteration gets the 5."""
        calls = []
        monkeypatch.setattr(paynt.dt.dtnest.api, "_run_dtnest", lambda *args: calls.append(args))
        task, build_task, factory = _dtnest_task_and_factory('P>=0.55 [F "goal"]')
        build_task.tree_depth = 2
        build_task.max_subtree_depth = 5
        paynt.dt.dtnest.synthesize(factory, task, build_task)
        [(_factory, _task, epsilon, subtree_depth, *_rest)] = calls
        assert subtree_depth == 5
        assert epsilon == build_task.error_threshold

    def test_loading_a_sketch_keeps_the_two_apart(self):
        factory, _task = load_sketch("tests/dt-orchard", task_kwargs={"tree_depth": 3, "max_subtree_depth": 5})
        assert (factory.build_task.tree_depth, factory.build_task.max_subtree_depth) == (3, 5)
        factory, _task = load_sketch("tests/dt-orchard", task_kwargs={"max_subtree_depth": 2})
        assert (factory.build_task.tree_depth, factory.build_task.max_subtree_depth) == (0, 2)
