from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import paynt.colored_mdp
import paynt.task
import paynt.result
import paynt.parameter_space.parameter_space
import paynt.synthesizer.statistic
import paynt.synthesizer.search_node
import paynt.utils.error_handling
import paynt.utils.timer

import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpaceEvaluation:
    """Result associated with a parameter space (subspace) after its evaluation."""

    parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
    value: Any
    sat: bool | None
    policy: Any
    selected_choices: Any = None


class Synthesizer:
    @staticmethod
    def for_method(colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask, method: str) -> Synthesizer:
        """Feature-agnostic dispatch: knows only the generic algorithms, never imports a specific feature package.

        AR and Hybrid are rejected for a colored MDP with a general coloring (see ColoredMdp.has_general_coloring).
        """
        # hiding imports here to avoid mutual top-level imports
        import paynt.synthesizer.synthesizer_onebyone
        import paynt.synthesizer.synthesizer_ar
        import paynt.synthesizer.synthesizer_cegis
        import paynt.synthesizer.synthesizer_hybrid
        import paynt.synthesizer.smpmc

        if method in ("ar", "hybrid"):
            paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, f"method {method!r}", "use --method onebyone, cegis or smpmc")

        if method == "onebyone":
            return paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(colored_mdp, task)
        if method == "ar":
            return paynt.synthesizer.synthesizer_ar.SynthesizerAR(colored_mdp, task)
        if method == "cegis":
            return paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS(colored_mdp, task)
        if method == "hybrid":
            return paynt.synthesizer.synthesizer_hybrid.SynthesizerHybrid(colored_mdp, task)
        if method == "smpmc":
            return paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task)
        raise ValueError("invalid method name")

    search_node_type = paynt.synthesizer.search_node.SearchNode

    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask):
        self.colored_mdp = colored_mdp
        self.task = task
        self.stat: paynt.synthesizer.statistic.Statistic | None = None
        self.synthesis_timer: paynt.utils.timer.Timer | None = None
        self.explored: int | None = None
        self.best_assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
        self.best_assignment_value: Any = None

    @property
    def method_name(self) -> str:
        """To be overridden."""
        raise NotImplementedError

    def time_limit_reached(self) -> bool:
        if (self.synthesis_timer is not None and self.synthesis_timer.time_limit_reached()) or paynt.utils.timer.GlobalTimer.time_limit_reached():
            logger.info("time limit reached, aborting...")
            return True
        return False

    def time_remaining(self) -> float | None:
        """Seconds until time_limit_reached() turns true (negative once it has), or None if there is no time limit."""
        synthesis_remaining = None if self.synthesis_timer is None else self.synthesis_timer.time_remaining()
        limits = [remaining for remaining in (synthesis_remaining, paynt.utils.timer.GlobalTimer.time_remaining()) if remaining is not None]
        return min(limits) if limits else None

    def memory_limit_reached(self) -> bool:
        if paynt.utils.timer.GlobalMemoryLimit.limit_reached():
            logger.info("memory limit reached, aborting...")
            return True
        return False

    def resource_limit_reached(self) -> bool:
        return self.time_limit_reached() or self.memory_limit_reached()

    def set_optimality_threshold(self, optimum_threshold: Any) -> None:
        if optimum_threshold is not None and self.task.specification.optimality is not None:
            self.task.specification.optimality.update_optimum(optimum_threshold)
            logger.debug(f"optimality threshold set to {optimum_threshold}")

    def explore(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> None:
        assert self.explored is not None
        self.explored += parameter_space.size

    def _reset_best_assignment(self) -> None:
        """Shared by synthesize() (when not keep_optimum) and run() (which always resets, but only after capturing best_assignment/best_assignment_value into
        the Result it returns)."""
        self.best_assignment = None
        self.best_assignment_value = None
        self.task.specification.reset()

    def evaluate_all(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, prop: Any, keep_value_only: bool = False) -> list[Any]:
        """To be overridden."""
        raise NotImplementedError

    def export_evaluation_result(self, evaluations: list[Any], export_filename_base: str) -> None:
        """To be overridden."""
        raise NotImplementedError

    def evaluate(
        self,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace | None = None,
        prop: Any = None,
        keep_value_only: bool = False,
        print_stats: bool = True,
    ) -> list[Any]:
        """Evaluate each member of the parameter space wrt the given property.

        :param parameter_space: if None, then the design space of the colored MDP will be used
        :param prop: if None, then the default property of the task will be used (assuming single-property specification)
        :param keep_value_only: if True, only value will be associated with the parameter space
        :param print_stats: if True, synthesis statistic will be printed
        :param export_filename_base: base filename used to export the evaluation results
        :returns: a list of (parameter_space,evaluation) pairs
        """
        if parameter_space is None:
            parameter_space = self.colored_mdp.parameter_space
        if prop is None:
            prop = self.task.get_property()

        self.stat = paynt.synthesizer.statistic.Statistic(self)
        self.explored = 0
        logger.info(f"evaluation initiated, design space: {parameter_space.size}")
        self.stat.start(parameter_space)
        evaluations = self.evaluate_all(parameter_space, prop, keep_value_only)
        self.stat.finished_evaluation(evaluations)
        logger.info("evaluation finished")

        if self.task.export_synthesis_filename_base is not None:
            self.export_evaluation_result(evaluations, self.task.export_synthesis_filename_base)

        if print_stats:
            self.stat.print()

        return evaluations

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        """To be overridden."""
        raise NotImplementedError

    def synthesize(
        self,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace | None = None,
        optimum_threshold: Any = None,
        keep_optimum: bool = False,
        return_all: bool = False,
        print_stats: bool = True,
        timeout: int | None = None,
    ) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        """
        :param parameter_space: parameter space (subspace) of assignments to search in
        :param optimum_threshold: known bound on the optimum value
        :param keep_optimum: if True, the optimality specification will not be reset upon finish
        :param return_all: if True and the synthesis returns a parameter space, all assignments will be
            returned instead of an arbitrary one
        :param print_stats: if True, synthesis stats will be printed upon completion
        :param timeout: synthesis time limit, seconds
        """
        if parameter_space is None:
            parameter_space = self.colored_mdp.parameter_space
        node = self.search_node_type(parameter_space)
        if node.constraint_indices is None:
            node.constraint_indices = list(range(len(self.task.specification.constraints)))

        self.set_optimality_threshold(optimum_threshold)
        self.synthesis_timer = paynt.utils.timer.Timer(timeout)
        self.synthesis_timer.start()
        self.stat = paynt.synthesizer.statistic.Statistic(self)
        self.explored = 0
        self.stat.start(parameter_space)
        self.synthesize_one(node)
        if self.best_assignment is not None and self.best_assignment.size > 1 and not return_all:
            self.best_assignment = self.best_assignment.pick_any()
        self.stat.finished_synthesis()
        if self.best_assignment is not None:
            logger.info("printing synthesized assignment below:")
            logger.info(self.best_assignment)
        elif optimum_threshold is not None and self.task.specification.optimality is not None:
            logger.info(
                f"no assignment improving the given --optimum-threshold {optimum_threshold} was found -- either {optimum_threshold} is already at or "
                "within model checking precision of the optimum, or no assignment reaches it at all"
            )

        if self.best_assignment is not None and self.best_assignment.size == 1:
            dtmc = self.colored_mdp.build_assignment(self.best_assignment)
            result = dtmc.check_specification(self.task.specification)
            logger.info(f"double-checking specification satisfiability: {result}")

        if print_stats:
            self.stat.print()

        assignment = self.best_assignment
        if not keep_optimum:
            self._reset_best_assignment()

        return assignment

    def run(self, optimum_threshold: Any = None) -> paynt.result.Result:
        assignment = self.synthesize(optimum_threshold=optimum_threshold, keep_optimum=True)
        value = self.best_assignment_value
        selected_choices = None
        if assignment is not None and assignment.size == 1:
            selected_choices = self.colored_mdp.selected_choices(assignment, self.task.specification)
        self._reset_best_assignment()
        return paynt.result.Result(success=assignment is not None, value=value, assignment=assignment, selected_choices=selected_choices)
