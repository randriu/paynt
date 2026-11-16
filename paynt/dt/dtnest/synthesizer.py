from __future__ import annotations

from typing import Any, cast

from .task import DtNestTask
from ..decision_tree import DecisionTree, DecisionTreeNode
from ..factory import DtColoredMdpFactory
from ..synthesizer import DtSynthesizer
from ..result import DtResult
from .._utils import DtInfo, DONT_CARE_ACTION_LABEL
import paynt.task
import paynt.utils.error_handling
import paynt.utils.timer
import paynt.model.model
from ._utils import (
    create_uniform_random_tree,
    get_submdp_from_unfixed_states,
    build_tree_helper_tree,
    get_state_space_for_tree_helper_node,
    run_scikit_learn_tree,
    state_to_choice_to_state_to_action,
    dt_to_state_to_actions,
    reduce_constraint_to_optimality,
    classify_constraint_threshold,
)

import stormpy

import logging

logger = logging.getLogger(__name__)


def _run_dtnest(
    cmdp_factory_dt: DtColoredMdpFactory,
    task: paynt.task.SynthesisTask,
    epsilon_error_threshold: float,
    max_subtree_depth: int,
    depth_fine_tuning: bool = True,
    allow_perturbations: bool = True,
    recompute_scheduler_perturbation: bool = True,
    timeout: int | None = 600,
) -> DtResult:
    synthesizer = DtNest(cmdp_factory_dt, task)
    synthesizer.epsilon = epsilon_error_threshold
    synthesizer.subtree_depth = max_subtree_depth
    synthesizer.depth_fine_tuning = depth_fine_tuning
    synthesizer.allow_perturbations = allow_perturbations
    synthesizer.recompute_scheduler = recompute_scheduler_perturbation
    synthesizer.timeout = timeout

    return synthesizer.run()


class DtNest(DtSynthesizer):
    def __init__(self, colored_mdp_factory: Any, task: paynt.task.SynthesisTask, *args: Any):
        paynt.utils.error_handling.require_no_step_bounds(
            task.specification,
            "--dtnest",
            "it searches with AR, which needs Storm's best scheduler of an MDP, and " + paynt.utils.error_handling.NO_SCHEDULER_FOR_STEP_BOUNDS,
            "use decision-tree synthesis with --method smpmc instead",
        )
        super().__init__(colored_mdp_factory, task, *args)
        self.best_tree: DecisionTree | None = None
        self.best_tree_value: Any = None

        # TODO add cli option to set the remaining settings below (max_subtree_depth/error_threshold/
        # initial_tree already flow from DtNestTask; everything else initialize_settings sets afterward
        # has no CLI knob yet)
        self.initialize_settings()

    def initialize_settings(self) -> None:
        # self.build_task is declared DtTask on the base DtSynthesizer, but DtNest always actually gets
        # a DtNestTask (max_subtree_depth/error_threshold are DtNestTask-only fields)
        build_task: DtNestTask = self.build_task  # type: ignore[assignment]
        self.subtree_depth = build_task.max_subtree_depth
        self.max_iter = 100000  # max number of subtrees to be investigated
        self.epsilon = build_task.error_threshold
        self.timeout = paynt.utils.timer.GlobalTimer.global_timer.time_limit_seconds if paynt.utils.timer.GlobalTimer.global_timer is not None else 600
        self.depth_fine_tuning = True  # decreases sub-tree depth once all subtrees of the current depth have been explored
        self.break_on_small_tree = True  # dtPAYNT synthesis ends when an implementable tree with good enough value is found
        self.allow_perturbations = True  # If False all perturbations are disabled, meaning we are only using dtPAYNT to make the initial tree smaller
        self.recompute_scheduler = True  # recomputes scheduler for the subtree outside of the replaced subtree
        self.use_states_for_node_priority = False  # this is super slow for some models but should mean better prioritization

    @property
    def method_name(self) -> str:
        return "dtNESt"

    def counters_reset(self) -> None:
        # integration stats
        self.dt_learning_calls = 0
        self.dt_learning_successes = 0
        self.dt_learning_recomputed_calls = 0
        self.dt_learning_recomputed_successes = 0
        self.dtpaynt_calls = 0
        self.dtpaynt_successes_smaller = 0
        self.dtpaynt_tree_found = 0
        self.all_larger = 0

    def choose_tree_to_use(
        self, current_tree: DecisionTree, dtpaynt_tree: DecisionTree, dtlearn_trees: dict[str, Any], recomputed_scheduler_trees: dict[str, Any]
    ) -> str:
        # this also defines the priority in case of a tie, therefore: current > dtpaynt > dtlearn > recomputed
        current_nodes = len(current_tree.collect_nonterminals())
        nodes = {"current": [current_nodes, current_tree.get_depth(), 1]}
        for setting, dtlearn_tree in recomputed_scheduler_trees.items():
            nodes["recomputed-" + setting] = [len(dtlearn_tree[1].collect_nonterminals()), dtlearn_tree[1].get_depth(), 1]
        for setting, dtlearn_tree in dtlearn_trees.items():
            nodes["dtlearn-" + setting] = [len(dtlearn_tree[1].collect_nonterminals()), dtlearn_tree[1].get_depth(), 1]
        nodes["dtpaynt"] = [len(dtpaynt_tree.collect_nonterminals()), dtpaynt_tree.get_depth(), 1]
        nodes = {k: v for k, v in nodes.items() if v is not None}
        sorted_nodes = sorted(nodes.items(), key=lambda item: item[1][1])
        sorted_nodes = sorted(nodes.items(), key=lambda item: item[1][0])

        logger.info(f"subtree information: {sorted_nodes}")
        return sorted_nodes[0][0]

    def create_tree_node_queue_heuristic(
        self, helper_tree: DecisionTree, desired_depth: int = 6, nodes_to_skip: list[int] | None = None, use_states_for_node_priority: bool = False
    ) -> list[dict[str, Any]]:
        if nodes_to_skip is None:
            nodes_to_skip = []
        nodes = helper_tree.collect_nodes(lambda node: node.get_depth() == desired_depth)
        if nodes is None or len(nodes) == 0:
            return []
        info = cast(DtInfo, self.colored_mdp.feature_info)
        helper_nodes = [info.tree_helper[node.identifier] for node in nodes]
        helper_node_stats = []
        for helper_node in helper_nodes:
            if helper_node["id"] == 0 or helper_node["id"] in nodes_to_skip:
                continue

            def matches_helper_node_id(candidate: DecisionTreeNode, target_id: int = helper_node["id"]) -> bool:
                return candidate.identifier == target_id

            helper_tree_node = helper_tree.collect_nodes(matches_helper_node_id)[0]
            if use_states_for_node_priority:
                stats = {
                    "id": helper_node["id"],
                    "states": get_state_space_for_tree_helper_node(self.colored_mdp, helper_node["id"]),
                    "nodes": helper_tree_node.get_number_of_descendants(),
                }
            else:
                stats = {"id": helper_node["id"], "nodes": helper_tree_node.get_number_of_descendants()}

            if "evaluations" not in helper_node.keys():
                stats["predicates"] = {}
            else:
                stats["predicates"] = {
                    pred: eval for pred, eval in helper_node["evaluations"].items() if eval <= list(helper_node["evaluations"].values())[0] * 1.05
                }
            helper_node_stats.append(stats)

        if len(helper_node_stats) == 0:
            return []

        helper_node_stats = sorted(helper_node_stats, key=lambda x: x["nodes"], reverse=True)
        helper_node_stats = sorted(helper_node_stats, key=lambda x: len(x["predicates"]), reverse=True)
        if use_states_for_node_priority:
            helper_node_stats = sorted(helper_node_stats, key=lambda x: x["states"].get_number_of_set_bits() / x["nodes"])

        return helper_node_stats

    @staticmethod
    def remap_node_queue_after_replacement(node_queue: list[dict[str, Any]], new_tree: DecisionTree) -> list[dict[str, Any]]:
        """After a subtree replacement, every surviving node_queue entry's "id" (an identifier in the OLD tree) must be translated to its counterpart in
        new_tree (nodes there carry old_identifier == their OLD-tree identifier, set by DecisionTreeNode.assign_identifiers(keep_old=True) right after the
        replacement)."""
        remapped_node_queue = []
        for node in node_queue:

            def matches_old_identifier(candidate: DecisionTreeNode, target_id: int = node["id"]) -> bool:
                return candidate.old_identifier == target_id

            nodes = new_tree.collect_nodes(matches_old_identifier)
            if len(nodes) == 0:
                logger.info(f"node {node['id']} was made obsolete by this replacement, dropping it from the queue")
                continue
            assert len(nodes) == 1, f"expected at most one node with old_identifier equal to {node['id']}, found {len(nodes)}"
            new_node = nodes[0]
            node["id"] = new_node.identifier
            remapped_node_queue.append(node)
        return remapped_node_queue

    def synthesize_subtrees(
        self,
        opt_result_value: float,
        random_result_value: float | None = None,
        user_threshold: float | None = None,
        user_threshold_minimizing: bool | None = None,
    ) -> None:

        # init
        self.counters_reset()
        info = cast(DtInfo, self.colored_mdp.feature_info)

        epsilon_already_derived = False
        if user_threshold is not None:
            # a bare constraint (e.g. P>=0.95) was reduced to an optimality query by run() -- DtNest has no
            # way to check a constraint directly, so its own epsilon-band acceptance test is repurposed here
            # to accept "clears the user's actual threshold" instead of "close to the true optimum"
            assert user_threshold_minimizing is not None
            classification, derived_epsilon = classify_constraint_threshold(opt_result_value, random_result_value, user_threshold, user_threshold_minimizing)
            if classification == "unsat":
                logger.warning(
                    f"the requested threshold {user_threshold} is not achievable -- the best possible value is {opt_result_value}; no admissible tree exists"
                )
                self.best_tree = None
                self.best_tree_value = None
                return
            if classification == "trivial":
                logger.info(
                    f"the requested threshold {user_threshold} is already satisfied by the random/don't-care "
                    f"scheduler (value {random_result_value}); returning it directly without subtree search"
                )
                self.best_tree = create_uniform_random_tree(info)
                self.best_tree_value = random_result_value
                return
            assert derived_epsilon is not None
            self.epsilon = derived_epsilon
            epsilon_already_derived = True

        if random_result_value is None:
            # if we dont have the random result value just make the threshold to be the epsilon of optimum value
            if not epsilon_already_derived:
                mc_result_positive = opt_result_value > 0
                assert self.task.specification.optimality is not None
                if self.task.specification.optimality.maximizing == mc_result_positive:
                    self.epsilon *= -1
            eps_optimum_threshold = opt_result_value * (1 + self.epsilon)
        else:  # this should result in normalised value of the produced tree being within espilon
            opt_random_diff = opt_result_value - random_result_value
            eps_optimum_threshold = opt_result_value - self.epsilon * opt_random_diff
        self.synthesis_timer = paynt.utils.timer.Timer(self.timeout)
        self.synthesis_timer.start()

        # initialize from external tree
        info.tree_helper_tree = build_tree_helper_tree(info)
        tree_helper_tree = info.tree_helper_tree
        logger.info(f"initial external tree has depth {tree_helper_tree.get_depth()} and {len(tree_helper_tree.collect_nonterminals())} nodes")

        current_iter = 0
        current_depth = self.subtree_depth

        while (self.depth_fine_tuning and current_depth > 1) or (current_depth == self.subtree_depth):
            if self.synthesis_timer.time_limit_reached():
                logger.info("timeout reached")
                break

            logger.info(f"starting iteration with subtree depth {current_depth}")
            # TODO this is not guaranteed to work in subsequent iterations when the dtPAYNT tree is used
            # I don't know what this comment means anymore but I couldn't reproduce any issues...
            node_queue = self.create_tree_node_queue_heuristic(
                tree_helper_tree, desired_depth=current_depth, use_states_for_node_priority=self.use_states_for_node_priority
            )

            while len(node_queue) > 0 and current_iter < self.max_iter:
                if self.synthesis_timer.time_limit_reached():
                    logger.info("timeout reached")
                    break

                logger.info(f"starting iteration {current_iter} with {len(node_queue)} nodes in node queue")
                logger.info(f"current tree size: {len(tree_helper_tree.collect_nonterminals())} decision nodes")

                current_iter += 1
                node = node_queue.pop(0)

                # subtree synthesis
                if self.use_states_for_node_priority:
                    node_states = node["states"]
                else:
                    node_states = get_state_space_for_tree_helper_node(self.colored_mdp, node["id"])
                submdp = get_submdp_from_unfixed_states(self.colored_mdp, node_states)
                logger.info(f"subtree MDP has {submdp.model.nr_states} states and {submdp.model.nr_choices} choices")
                subtree_spec = self.task.specification.copy()
                subtree_task = paynt.task.SynthesisTask.from_specification(subtree_spec, use_exact=self.colored_mdp.use_exact)
                subtree_colored_mdp_factory = DtColoredMdpFactory(submdp.model, self.colored_mdp_factory.build_task)
                assert subtree_task.specification.optimality is not None
                subtree_task.specification.optimality.update_optimum(eps_optimum_threshold)
                # initial_depth=0: this construction's own tree is only ever used to inspect depth-invariant
                # feature_info below (state_is_relevant_bv/action_labels, fixed once in the factory's own
                # __init__) -- building at a larger depth here would be needlessly expensive and, in the
                # synthesize_tree_sequence branch below, immediately discarded and rebuilt from depth 0 anyway
                subtree_synthesizer = DtSynthesizer(subtree_colored_mdp_factory, subtree_task, initial_depth=0)
                self.dtpaynt_calls += 1

                subtree_info = cast(DtInfo, subtree_synthesizer.colored_mdp.feature_info)
                if subtree_info.state_is_relevant_bv.get_number_of_set_bits() == 0:
                    random_tree = create_uniform_random_tree(subtree_info)
                    subtree_synthesizer.best_tree = random_tree
                else:
                    subtree_synthesizer.synthesize_tree_sequence(
                        opt_result_value, overall_timeout=60, max_depth=current_depth, break_if_found=self.break_on_small_tree
                    )

                # create new tree
                if subtree_synthesizer.best_tree is not None:
                    logger.info(f"admissible subtree found from node {node['id']}")
                    self.dtpaynt_tree_found += 1
                    relevant_state_valuations = [subtree_info.relevant_state_valuations[state] for state in subtree_info.state_is_relevant_bv]
                    subtree_synthesizer.best_tree.simplify(relevant_state_valuations)
                    dtpaynt_subtree_helper_tree_copy = tree_helper_tree.copy()
                    dtpaynt_subtree_helper_tree_copy.append_tree_as_subtree(subtree_synthesizer.best_tree, node["id"], subtree_info)
                    dtpaynt_subtree_helper_tree_copy.root.assign_identifiers(keep_old=True)
                    new_tree_depth = dtpaynt_subtree_helper_tree_copy.get_depth()
                    new_tree_nodes = len(dtpaynt_subtree_helper_tree_copy.collect_nonterminals())
                    logger.info(f"new tree has depth {new_tree_depth} and {new_tree_nodes} nodes")

                    info.tree_helper_tree = dtpaynt_subtree_helper_tree_copy

                    new_tree_helper_tree = None
                    recomputed_scheduler_tree_helper_tree = None
                    dtlearn_trees = {}
                    recomputed_dtlearn_trees = {}

                    if self.allow_perturbations:
                        submdp_for_tree = get_submdp_from_unfixed_states(self.colored_mdp, node_states)
                        reachable_states = stormpy.datastructures.BitVector(self.colored_mdp.underlying_mdp.nr_states, False)
                        for state in range(submdp_for_tree.model.nr_states):
                            reachable_states.set(submdp_for_tree.underlying_mdp_state_map[state], True)

                        # Perturbation #1 - learn new tree based on the scheduler of the new updated tree
                        state_to_action = dt_to_state_to_actions(dtpaynt_subtree_helper_tree_copy, self.colored_mdp, reachable_states)
                        new_learned_tree_helper = run_scikit_learn_tree(info.relevant_state_valuations, state_to_action, info.variables, info.action_labels)
                        self.dt_learning_calls += 1

                        new_tree_helper_tree = build_tree_helper_tree(info, new_learned_tree_helper)
                        learned_tree_depth = new_tree_helper_tree.get_depth()
                        learned_tree_nodes = len(new_tree_helper_tree.collect_nonterminals())
                        logger.info(f"new learned tree (default) has depth {learned_tree_depth} and {learned_tree_nodes} nodes")
                        dtlearn_trees["default"] = (new_learned_tree_helper, new_tree_helper_tree)

                        #  Perturbations #2 - recompute scheduler for states outside of the new subtree and learn new tree based on that scheduler
                        if self.recompute_scheduler:
                            submpd_dtlearn_of_subtree = get_submdp_from_unfixed_states(self.colored_mdp, ~node_states)
                            oos_result = submpd_dtlearn_of_subtree.check_specification(self.task.specification)
                            new_scheduler = oos_result.optimality_result.result.get_scheduler()
                            state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(
                                self.colored_mdp.underlying_mdp, self.colored_mdp.choice_destinations, submpd_dtlearn_of_subtree, new_scheduler
                            )

                            recomputed_state_to_action = state_to_choice_to_state_to_action(state_to_choice, self.colored_mdp)
                            recomputed_scheduler_tree_helper = run_scikit_learn_tree(
                                info.relevant_state_valuations,
                                recomputed_state_to_action,
                                info.variables,
                                info.action_labels,
                            )
                            self.dt_learning_recomputed_calls += 1

                            recomputed_scheduler_tree_helper_tree = build_tree_helper_tree(info, recomputed_scheduler_tree_helper)
                            recomputed_tree_depth = recomputed_scheduler_tree_helper_tree.get_depth()
                            recomputed_tree_nodes = len(recomputed_scheduler_tree_helper_tree.collect_nonterminals())
                            logger.info(
                                f"new learned tree (default) based on recomputed scheduler has depth {recomputed_tree_depth} and {recomputed_tree_nodes} nodes"
                            )

                            recomputed_dtlearn_trees["default"] = (recomputed_scheduler_tree_helper, recomputed_scheduler_tree_helper_tree)

                    chosen_tree = self.choose_tree_to_use(tree_helper_tree, dtpaynt_subtree_helper_tree_copy, dtlearn_trees, recomputed_dtlearn_trees)

                    if chosen_tree == "current":
                        logger.info("None of the new trees are smaller, continuing with current tree")
                        self.all_larger += 1
                        info.tree_helper_tree = tree_helper_tree

                    elif chosen_tree == "dtpaynt":
                        logger.info("New dtPAYNT tree is smallest")
                        self.dtpaynt_successes_smaller += 1
                        tree_helper_tree = dtpaynt_subtree_helper_tree_copy
                        info.tree_helper_tree = tree_helper_tree
                        node_queue = self.remap_node_queue_after_replacement(node_queue, tree_helper_tree)
                        new_nodes = self.create_tree_node_queue_heuristic(
                            tree_helper_tree,
                            desired_depth=current_depth,
                            nodes_to_skip=[node["id"] for node in node_queue],
                            use_states_for_node_priority=self.use_states_for_node_priority,
                        )
                        node_queue += new_nodes

                    elif chosen_tree.startswith("dtlearn"):
                        logger.info(f"New learned tree ({chosen_tree}) is smallest")
                        dtlearn_setting = chosen_tree.split("-")[1]
                        new_dtlearn_tree_helper = dtlearn_trees[dtlearn_setting][0]
                        new_dtlearn_tree_helper_tree = dtlearn_trees[dtlearn_setting][1]
                        self.dt_learning_successes += 1
                        info.tree_helper = new_dtlearn_tree_helper
                        info.tree_helper_tree = new_dtlearn_tree_helper_tree
                        tree_helper_tree = new_dtlearn_tree_helper_tree
                        node_queue = self.create_tree_node_queue_heuristic(tree_helper_tree, use_states_for_node_priority=self.use_states_for_node_priority)

                    elif chosen_tree.startswith("recomputed"):
                        logger.info(f"New learned tree ({chosen_tree}) for recomputed scheduler is smallest")
                        dtlearn_setting = chosen_tree.split("-")[1]
                        recomputed_scheduler_tree_helper = recomputed_dtlearn_trees[dtlearn_setting][0]
                        recomputed_scheduler_tree_helper_tree = recomputed_dtlearn_trees[dtlearn_setting][1]
                        self.dt_learning_recomputed_successes += 1
                        info.tree_helper = recomputed_scheduler_tree_helper
                        info.tree_helper_tree = recomputed_scheduler_tree_helper_tree
                        tree_helper_tree = recomputed_scheduler_tree_helper_tree
                        node_queue = self.create_tree_node_queue_heuristic(tree_helper_tree, use_states_for_node_priority=self.use_states_for_node_priority)

                else:
                    logger.info(f"no admissible subtree found from node {node['id']}")

            current_depth -= 1

        info.tree_helper_tree = tree_helper_tree

        self.synthesis_timer.stop()

        # double check
        dtmc = get_submdp_from_unfixed_states(self.colored_mdp)
        # TODO why would it not be a DTMC here???
        if dtmc.model.nr_states != dtmc.model.nr_choices:
            logger.info("tree did not induce dtmc?")
        # assert dtmc.model.nr_states == dtmc.model.nr_choices, "tree did not induce dtmc"
        result = dtmc.check_specification(self.task.specification)

        # TODO find out why this would not hold????
        if opt_result_value < eps_optimum_threshold:
            assert result.optimality_result.value <= eps_optimum_threshold, (
                f"optimum value {result.optimality_result.value} is not below threshold {eps_optimum_threshold}"
            )
        else:
            assert result.optimality_result.value >= eps_optimum_threshold, (
                f"optimum value {result.optimality_result.value} is not above threshold {eps_optimum_threshold}"
            )

        self.best_tree = info.tree_helper_tree
        self.best_tree_value = result.optimality_result.value

        final_tree_depth = info.tree_helper_tree.get_depth()
        final_tree_nodes = len(info.tree_helper_tree.collect_nonterminals())
        logger.info(f"final tree has value {result.optimality_result.value} with depth {final_tree_depth} and {final_tree_nodes} nodes")

    def run(self, optimum_threshold: Any = None) -> DtResult:
        # see initialize_settings's comment: DtNest always actually gets a DtNestTask
        build_task: DtNestTask = self.build_task  # type: ignore[assignment]
        info = cast(DtInfo, self.colored_mdp.feature_info)

        paynt_mdp = paynt.model.model.SubMdp(
            self.colored_mdp.underlying_mdp, list(range(self.colored_mdp.underlying_mdp.nr_states)), list(range(self.colored_mdp.underlying_mdp.nr_choices))
        )

        # DtNest always works from a single optimality-shaped query (see synthesize_subtrees): it has no
        # mechanism to check a constraint directly, only to approximate a numeric target within an
        # epsilon-band. A specification with no optimality objective but exactly one constraint is still
        # usable -- reduce_constraint_to_optimality converts it, and its original threshold flows into
        # synthesize_subtrees so DtNest's own acceptance test targets exactly what the user asked for
        # (see classify_constraint_threshold) instead of silently chasing the unconstrained optimum.
        user_threshold: float | None = None
        user_threshold_minimizing: bool | None = None
        spec = self.task.specification
        if spec.optimality is not None:
            if len(spec.constraints) > 0:
                raise ValueError(
                    f"dtnest does not support constraints alongside an optimality objective (found "
                    f"{len(spec.constraints)} constraint(s)) -- it has no mechanism to check a constraint, "
                    f"only to approximate a single optimality query"
                )
        elif len(spec.constraints) == 1:
            self.task.specification, constraint = reduce_constraint_to_optimality(spec)
            user_threshold = constraint.threshold
            user_threshold_minimizing = constraint.minimizing
            logger.info(
                f"no optimality objective given -- treating the constraint {constraint} as an optimality "
                f"objective for dtnest, using its threshold {user_threshold} as the acceptance target"
            )
        else:
            raise ValueError(
                f"dtnest requires a specification with either an optimality objective or exactly one "
                f"constraint to approximate as one; found {len(spec.constraints)} constraint(s) and no "
                f"optimality objective"
            )

        mc_result = paynt_mdp.model_check_property(self.task.get_property())
        opt_scheduler = mc_result.result.get_scheduler()

        state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(
            self.colored_mdp.underlying_mdp, self.colored_mdp.choice_destinations, paynt_mdp, opt_scheduler
        )

        if build_task.initial_tree is None:
            state_to_action = state_to_choice_to_state_to_action(state_to_choice, self.colored_mdp)
            initial_tree_helper = run_scikit_learn_tree(info.relevant_state_valuations, state_to_action, info.variables, info.action_labels)

        else:
            # TODO add some nice export for trees, decide on the format we will support here
            raise NotImplementedError("the support for user provided initial tree is not implemented.")

        info.tree_helper = initial_tree_helper

        opt_result_value = mc_result.value
        logger.info(f"the optimal scheduler has value: {opt_result_value}")

        if DONT_CARE_ACTION_LABEL in info.action_labels:
            from .._utils import get_random_choices

            random_choices = get_random_choices(self.colored_mdp)
            submdp_random = self.colored_mdp.build_from_choice_mask(random_choices)
            mc_result_random = submdp_random.model_check_property(self.task.get_property())
            random_result_value = mc_result_random.value
            logger.info(f"the random scheduler has value: {random_result_value}")
            # self.set_optimality_threshold(random_result_value)
        else:
            # no synthetic don't-care action to build a random-scheduler baseline from (e.g.
            # add_dont_care_action=False) -- fall back to a genuine, sound baseline instead of leaving
            # random_result_value unbound: the value the worst-case scheduler achieves over the full
            # underlying MDP.
            mc_result_worst = paynt_mdp.model_check_property(self.task.get_property(), alt=True)
            random_result_value = mc_result_worst.value
            logger.info(f"no don't-care action available; using the worst-case scheduler value as a baseline: {random_result_value}")

        self.best_tree = self.best_tree_value = None

        assert info.tree_helper is not None, "tree helper not set, cannot run dtNest"

        self.synthesize_subtrees(opt_result_value, random_result_value, user_threshold, user_threshold_minimizing)

        logger.info(f"the optimal scheduler has value: {opt_result_value}")
        logger.info(f"the baseline (random or worst-case) scheduler has value: {random_result_value}")
        if self.best_tree is None:
            logger.info("no admissible tree found")
        else:
            relevant_state_valuations = [info.relevant_state_valuations[state] for state in info.state_is_relevant_bv]
            self.best_tree.simplify(relevant_state_valuations)
            depth = self.best_tree.get_depth()
            num_nodes = len(self.best_tree.collect_nonterminals())
            logger.info(f"synthesized tree of depth {depth} with {num_nodes} decision nodes")
            if self.task.specification.has_optimality:
                logger.info(f"the synthesized tree has value {self.best_tree_value}")
                logger.info(
                    f"the synthesized tree has relative value: {self.compute_normalized_value(self.best_tree_value, opt_result_value, random_result_value)}"
                )
            logger.info("printing the synthesized tree below:")

            # integration logs
            if info.tree_helper is not None:
                logger.info(f"dt learning calls: {self.dt_learning_calls}")
                logger.info(f"dt learning successes: {self.dt_learning_successes}")
                logger.info(f"dt learning recomputed calls: {self.dt_learning_recomputed_calls}")
                logger.info(f"dt learning recomputed successes: {self.dt_learning_recomputed_successes}")
                logger.info(f"dtpaynt calls: {self.dtpaynt_calls}")
                logger.info(f"dtpaynt successes smaller: {self.dtpaynt_successes_smaller}")
                logger.info(f"dtpaynt tree found: {self.dtpaynt_tree_found}")
                logger.info(f"all larger: {self.all_larger}")

            if self.task.export_synthesis_filename_base is not None:
                self.export_decision_tree(self.best_tree, self.task.export_synthesis_filename_base)

        return DtResult(success=self.best_tree is not None, value=self.best_tree_value, tree=self.best_tree)
