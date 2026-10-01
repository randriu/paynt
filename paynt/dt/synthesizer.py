from __future__ import annotations

from typing import Any, cast

import paynt.task
import paynt.dt.task
import paynt.dt.factory
import paynt.dt._utils
import paynt.dt.decision_tree
import paynt.synthesizer.statistic
import paynt.utils.error_handling
import paynt.utils.timer
import paynt.model.model

import paynt.dt.result
from paynt.dt.synthesizer_ar_dt import SynthesizerARDt

from ._utils import DT_INNER_METHODS, make_inner_synthesizer, simplify_tree

import payntbind

import os
import json

import logging

logger = logging.getLogger(__name__)


def _choose_solver_for_dt_task(build_task: paynt.dt.task.DtTask) -> str:
    if build_task.has_scheduler_to_map:
        return "dtmap"
    return "dtpaynt"


def _run_dt_map_scheduler(
    cmdp_factory_dt: paynt.dt.factory.DtColoredMdpFactory, task: paynt.task.SynthesisTask, scheduler: Any, tree_depth: int, method: str = "ar"
) -> paynt.dt.result.DtResult:
    """Helper function to map a scheduler to a decision tree using the DTMap algorithm.

    Returns a tuple (success, decision_tree).

    :param method: see DtSynthesizer; only "ar" maps a scheduler
    """
    state_to_choice = payntbind.synthesis.schedulerToStateToGlobalChoice(
        scheduler, cmdp_factory_dt.underlying_mdp, list(range(cmdp_factory_dt.underlying_mdp.nr_choices))
    )
    state_to_choice = paynt.model.model.ModelIndex.discard_unreachable_choices(
        cmdp_factory_dt.underlying_mdp, cmdp_factory_dt.choice_destinations, state_to_choice
    )
    choices = paynt.model.model.ModelIndex.state_to_choice_to_choices(cmdp_factory_dt.underlying_mdp, state_to_choice)

    dt_synthesizer = DtSynthesizer(cmdp_factory_dt, task, method=method)
    dt_synthesizer.map_scheduler(choices, tree_depth=tree_depth)

    simplify_tree(dt_synthesizer.best_tree, dt_synthesizer.colored_mdp.feature_info)

    return paynt.dt.result.DtResult(success=dt_synthesizer.best_tree is not None, value=dt_synthesizer.best_tree_value, tree=dt_synthesizer.best_tree)


def _run_dtpaynt(
    cmdp_factory_dt: paynt.dt.factory.DtColoredMdpFactory, task: paynt.task.SynthesisTask, tree_depth: int, timeout: int | None = None, method: str = "ar"
) -> paynt.dt.result.DtResult:
    dt_synthesizer = DtSynthesizer(cmdp_factory_dt, task, method=method)
    dt_synthesizer.synthesize_tree(tree_depth, timeout=timeout)

    simplify_tree(dt_synthesizer.best_tree, dt_synthesizer.colored_mdp.feature_info)

    return paynt.dt.result.DtResult(success=dt_synthesizer.best_tree is not None, value=dt_synthesizer.best_tree_value, tree=dt_synthesizer.best_tree)


class DtSynthesizer:
    """Outer driver: repeatedly re-unfolds the decision tree at different depths (DtColoredMdpFactory.reset_tree tries a fresh depth/coloring each time, unlike
    the FSC-unfolding factories' memory-size growth) and runs an inner search engine -- a fresh instance constructed for each depth -- against each unfolding,
    keeping the best tree found so far across depths.

    :param method: the inner engine -- "ar" (default), SynthesizerARDt over ColoringSmt, dtpaynt's own unchanged default path; or "smpmc", SynthesizerSMPMC over
        the tree's ColoringGeneral (see paynt.dt.coloring_general). Mapping a scheduler (map_scheduler) supports only "ar".
    """

    def __init__(
        self,
        colored_mdp_factory: paynt.dt.factory.DtColoredMdpFactory,
        task: paynt.task.SynthesisTask,
        initial_depth: int | None = None,
        method: str = "ar",
    ):
        paynt.utils.error_handling.require_supported_method(method, DT_INNER_METHODS, "decision-tree synthesis")
        self.colored_mdp_factory = colored_mdp_factory
        self.task = task
        self.method = method
        # build_task is Optional at the type level (a factory can be constructed before its DtTask is known),
        # but DtSynthesizer always needs one already attached by construction time; cached here since
        # run()/synthesize_tree_sequence read DT-specific fields constantly.
        assert colored_mdp_factory.build_task is not None
        self.build_task: paynt.dt.task.DtTask = colored_mdp_factory.build_task
        if self.build_task.scheduler_path is not None or self.build_task.has_scheduler_to_map:
            paynt.utils.error_handling.require_method_ar(method, "mapping a scheduler to a tree")
        # the factory never builds automatically -- request the initial tree explicitly. Defaults to
        # build_task's own depth; callers only needing cheap, depth-invariant inspection (e.g. DtNest's
        # subtree recursion) pass initial_depth=0 to avoid building at a larger, more expensive depth than
        # necessary -- whatever gets built here is discarded and rebuilt from depth 0 once synthesis runs.
        self.colored_mdp = colored_mdp_factory.reset_tree(initial_depth if initial_depth is not None else self.build_task.tree_depth, general=(method != "ar"))
        self.best_tree: paynt.dt.decision_tree.DecisionTree | None = None
        self.best_tree_value: Any = None

    @property
    def method_name(self) -> str:
        return "AR (decision tree)" if self.method == "ar" else f"{self.method} (decision tree)"

    def compute_normalized_value(self, value: float, opt: float, random: float) -> float:
        return (value - random) / (opt - random) if opt - random != 0 else 1.0

    def export_decision_tree(self, decision_tree: paynt.dt.decision_tree.DecisionTree, export_filename_base: str) -> None:
        tree = decision_tree.to_graphviz()
        tree_filename = export_filename_base + ".dot"
        directory = os.path.dirname(tree_filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(tree_filename, "w") as file:
            file.write(tree.source)
        logger.info(f"exported decision tree to {tree_filename}")

        tree_visualization_filename = export_filename_base + ".png"
        tree.render(export_filename_base, format="png", cleanup=True)  # using export_filename_base since graphviz appends .png by default
        logger.info(f"exported decision tree visualization to {tree_visualization_filename}")

        tree_string_filename = export_filename_base + ".txt"
        with open(tree_string_filename, "w") as file:
            file.write(decision_tree.to_string())
        logger.info(f"exported decision tree string to {tree_string_filename}")

    def synthesize_tree(self, depth: int, timeout: int | None = None) -> None:
        self.colored_mdp = self.colored_mdp_factory.reset_tree(depth, general=(self.method != "ar"))
        synthesizer = make_inner_synthesizer(self.method, self.colored_mdp, self.task)
        synthesizer.synthesize(keep_optimum=True, timeout=timeout)
        if synthesizer.best_assignment is not None:
            info = cast(paynt.dt._utils.DtInfo, self.colored_mdp.feature_info)
            info.decision_tree.root.associate_assignment(synthesizer.best_assignment)
            self.best_tree = info.decision_tree
            self.best_tree_value = synthesizer.best_assignment_value

    def synthesize_tree_sequence(
        self, opt_result_value: float, overall_timeout: float | None = None, max_depth: int | None = None, break_if_found: bool = False
    ) -> None:
        self.best_tree = self.best_tree_value = None

        if max_depth is None:
            max_depth = self.build_task.tree_depth + 1
        if overall_timeout is None:
            assert paynt.utils.timer.GlobalTimer.global_timer is not None
            global_timeout = paynt.utils.timer.GlobalTimer.global_timer.time_limit_seconds
            if global_timeout is None:
                # TODO this should probably not be the default behaviour, we want to run the synthesis
                # indefinitely if the user does not give us a timeout
                global_timeout = 900
            overall_timeout = global_timeout
            tree_sequence_timer = None
        else:
            tree_sequence_timer = paynt.utils.timer.Timer(overall_timeout)
            tree_sequence_timer.start()
        depth_timeout = overall_timeout / 2 / (max_depth - 1) if max_depth > 1 else overall_timeout
        best_assignment: Any = None
        for depth in range(max_depth):
            self.colored_mdp = self.colored_mdp_factory.reset_tree(depth, general=(self.method != "ar"))
            info = cast(paynt.dt._utils.DtInfo, self.colored_mdp.feature_info)
            synthesizer = make_inner_synthesizer(self.method, self.colored_mdp, self.task)
            best_assignment_old = best_assignment

            parameter_space = self.colored_mdp.parameter_space
            node = synthesizer.search_node_type(parameter_space)
            synthesizer.explored = 0
            synthesizer.stat = paynt.synthesizer.statistic.Statistic(synthesizer)
            synthesizer.stat.start(parameter_space)
            timeout = depth_timeout if depth < max_depth - 1 else overall_timeout / 2  # second half of the time for the last depth
            synthesizer.synthesis_timer = paynt.utils.timer.Timer(timeout)
            synthesizer.synthesis_timer.start()
            nodes = [node]

            if self.best_tree is not None:
                parameter_subspace = parameter_space.copy()
                info.decision_tree.root.apply_hint(parameter_subspace, self.best_tree.root)
                nodes = [synthesizer.search_node_type(parameter_subspace), node]

            for n in nodes:
                synthesizer.synthesize_one(n)
            synthesizer.stat.finished_synthesis()
            synthesizer.stat.print()
            synthesizer.synthesis_timer = None

            best_assignment = synthesizer.best_assignment
            # best_assignment != best_assignment_old alone used to also fire when this depth found nothing
            # (best_assignment flipping from a real ParameterSpace to None), which then crashed below trying
            # to associate a None assignment onto self.best_tree (stale from an earlier, better depth)
            new_assignment_synthesized = best_assignment is not None and best_assignment != best_assignment_old
            if new_assignment_synthesized:
                logger.info("printing synthesized assignment below:")
                logger.info(best_assignment)

                if best_assignment is not None and best_assignment.size == 1:
                    dtmc = self.colored_mdp.build_assignment(best_assignment)
                    result = dtmc.check_specification(self.task.specification)
                    logger.info(f"double-checking specification satisfiability: {result}")

                self.best_tree = info.decision_tree
                self.best_tree.root.associate_assignment(best_assignment)
                self.best_tree_value = synthesizer.best_assignment_value

                if (
                    break_if_found
                    or (opt_result_value != 0 and abs((synthesizer.best_assignment_value - opt_result_value) / opt_result_value) < 1e-3)
                    or (opt_result_value == 0 and synthesizer.best_assignment_value < 1e-3)
                ):
                    break

            if synthesizer.resource_limit_reached() or tree_sequence_timer is not None and tree_sequence_timer.time_limit_reached():
                break

    def map_scheduler(self, scheduler_choices: Any, tree_depth: int | None = None) -> None:
        """Find the shallowest tree, up to tree_depth, that makes exactly the given scheduler's choices.

        Each depth is one satisfiability query, whether some tree of that depth is consistent with the scheduler, answered by ColoringSmt (through
        SynthesizerARDt, which also checks the tree found). There is no search, so no inner engine other than AR is involved.
        """
        if tree_depth is None:
            tree_depth = self.build_task.tree_depth
        for depth in range(tree_depth + 1):
            self.colored_mdp = self.colored_mdp_factory.reset_tree(depth, enable_harmonization=False)
            synthesizer = SynthesizerARDt(self.colored_mdp, self.task)
            node = synthesizer.search_node_type(self.colored_mdp.parameter_space.copy())
            node.analysis_result = synthesizer.build_unsat_result()
            node.mdp, node.selected_choices = self.colored_mdp.build(node.parameter_space)
            consistent, parameter_selection = self.colored_mdp.are_choices_consistent(scheduler_choices, node.parameter_space)
            if consistent:
                synthesizer.verify_parameter_selection(node, parameter_selection)
                if synthesizer.best_assignment is not None:
                    info = cast(paynt.dt._utils.DtInfo, self.colored_mdp.feature_info)
                    self.best_tree = info.decision_tree
                    self.best_tree.root.associate_assignment(synthesizer.best_assignment)
                    self.best_tree_value = synthesizer.best_assignment_value
                    break

            if synthesizer.resource_limit_reached():
                break

    def run(self, optimum_threshold: Any = None) -> paynt.dt.result.DtResult:
        scheduler_choices = None
        if self.build_task.scheduler_path is None:
            paynt_mdp = paynt.model.model.Mdp(self.colored_mdp.underlying_mdp)
            mc_result = paynt_mdp.model_check_property(self.task.get_property())
        else:
            opt_result_value = None
            with open(self.build_task.scheduler_path) as f:
                scheduler_json = json.load(f)
            scheduler_choices, scheduler_json_relevant = paynt.dt._utils.scheduler_json_to_choices(
                self.colored_mdp, scheduler_json, discard_unreachable_states=True
            )

            submdp = self.colored_mdp.build_from_choice_mask(scheduler_choices)
            mc_result = submdp.model_check_property(self.task.get_property())
        opt_result_value = mc_result.value
        logger.info(f"the optimal scheduler has value: {opt_result_value}")

        info = cast(paynt.dt._utils.DtInfo, self.colored_mdp.feature_info)
        if paynt.dt._utils.DONT_CARE_ACTION_LABEL in info.action_labels:
            random_choices = paynt.dt._utils.get_random_choices(self.colored_mdp)
            submdp_random = self.colored_mdp.build_from_choice_mask(random_choices)
            mc_result_random = submdp_random.model_check_property(self.task.get_property())
            random_result_value = mc_result_random.value
            logger.info(f"the random scheduler has value: {random_result_value}")

        self.best_tree = self.best_tree_value = None
        if scheduler_choices is not None:
            self.map_scheduler(scheduler_choices)
        else:
            if self.task.specification.optimality is not None:
                epsilon = 1e-1
                mc_result_positive = opt_result_value > 0
                if self.task.specification.optimality.maximizing == mc_result_positive:
                    epsilon *= -1
            # equivalent to Synthesizer.set_optimality_threshold, inlined since this outer driver isn't a
            # Synthesizer subclass itself (only the inner SynthesizerARDt engines it constructs are)
            if self.task.specification.optimality is not None and optimum_threshold is not None:
                self.task.specification.optimality.update_optimum(optimum_threshold)

            if not self.build_task.tree_enumeration:
                self.synthesize_tree(self.build_task.tree_depth)
            else:
                self.synthesize_tree_sequence(opt_result_value)

        logger.info(f"the optimal scheduler has value: {opt_result_value}")
        # self.colored_mdp may have been reassigned by map_scheduler/synthesize_tree(_sequence) above, so
        # re-fetch info -- action_labels/relevant_state_valuations/state_is_relevant_bv are the same MDP
        # identity across every reset_tree() call regardless, but re-fetching avoids relying on that
        info = cast(paynt.dt._utils.DtInfo, self.colored_mdp.feature_info)
        if paynt.dt._utils.DONT_CARE_ACTION_LABEL in info.action_labels:
            logger.info(f"the random scheduler has value: {random_result_value}")
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
                if paynt.dt._utils.DONT_CARE_ACTION_LABEL in info.action_labels:
                    logger.info(
                        f"the synthesized tree has relative value: {self.compute_normalized_value(self.best_tree_value, opt_result_value, random_result_value)}"
                    )
            logger.info(f"printing the synthesized tree below:\n{self.best_tree.to_string()}")

            if self.task.export_synthesis_filename_base is not None:
                self.export_decision_tree(self.best_tree, self.task.export_synthesis_filename_base)

        time_total = round(paynt.utils.timer.GlobalTimer.read(), 2)
        logger.info(f"synthesis finished after {time_total} seconds")

        return paynt.dt.result.DtResult(success=self.best_tree is not None, value=self.best_tree_value, tree=self.best_tree)
