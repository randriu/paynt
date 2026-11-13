from __future__ import annotations

import math
from typing import Any

import paynt.colored_mdp
import paynt.task
import paynt.parameter_space.parameter_space
import paynt.synthesizer.synthesizer
import paynt.synthesizer.search_node
import paynt.specification.property_result
import paynt.model.model
import paynt.utils.scoring

import logging

logger = logging.getLogger(__name__)


def scheduler_scores(
    colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask, mdp: Any, prop: Any, result: Any, selection: list[list[int]]
) -> dict[int, float] | None:
    inconsistent_assignments = {parameter: options for parameter, options in enumerate(selection) if len(options) > 1}
    choice_values = paynt.model.model.ModelIndex.choice_values(mdp.model, prop, result.get_value_vector())
    choices = result.get_scheduler().compute_action_support(mdp.model.nondeterministic_choice_indices)
    expected_visits = paynt.model.model.ModelIndex.compute_expected_visits(mdp.model, prop, choices, disable_expected_visits=task.disable_expected_visits)
    # POMDP has a specialized, hand-optimized scorer for the common posterior-unaware case; every other
    # colored-MDP variant (and posterior-aware POMDPs) uses the generic implementation. Dispatched by
    # feature_kind, not an isinstance check, so this module never needs to import paynt.pomdp.
    if colored_mdp.feature_kind == "pomdp" and not colored_mdp.feature_info.posterior_aware:
        scores = paynt.utils.scoring.estimate_scheduler_difference_pomdp(
            colored_mdp, mdp.model, mdp.underlying_mdp_choice_map, inconsistent_assignments, choice_values, expected_visits
        )
    else:
        scores = paynt.utils.scoring.estimate_scheduler_difference(
            colored_mdp, mdp.model, mdp.underlying_mdp_choice_map, inconsistent_assignments, choice_values, expected_visits
        )
    return scores


def scheduler_scores_combined(colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask, mdp: Any, results: list[Any]) -> dict[int, float]:
    """Aggregate v(h) across every still-relevant property (MdpSpecificationResult.undecided_results()), taking the max per parameter across properties that
    consider it inconsistent.

    A property whose own selection is already fully consistent contributes nothing here (it's an L(h) candidate instead, handled by the caller).
    """
    combined: dict[int, float] = {}
    for result in results:
        if all(len(options) <= 1 for options in result.primary_selection):
            continue
        scores = scheduler_scores(colored_mdp, task, mdp, result.prop, result.primary.result, result.primary_selection)
        assert scores is not None
        for parameter, score in scores.items():
            if parameter not in combined or score > combined[parameter]:
                combined[parameter] = score
    return combined


def set_design_apart(colored_mdp: paynt.colored_mdp.ColoredMdp, node: paynt.synthesizer.search_node.SearchNode, result: Any) -> tuple[int, list[list[int]]]:
    """(the splitter, its suboptions) that separate the design of a consistent scheduler from the others on one parameter it uses.

    The designs that agree with it on every parameter it uses behave like it, so each such split brings it closer to a node of its own.
    """
    assert node.mdp is not None
    used = colored_mdp.scheduler_selection(node.mdp, result.primary.result.get_scheduler())
    splitters = [parameter for parameter, options in enumerate(used) if options and node.parameter_space.parameter_num_options(parameter) > 1]
    if not splitters:
        # every design of the node agrees with it where it matters, though the model has not collapsed into one design yet
        splitter = next(parameter for parameter in range(node.parameter_space.num_parameters) if node.parameter_space.parameter_num_options(parameter) > 1)
        return splitter, node.parameter_space.suboptions_half(splitter)
    splitter = splitters[0]
    [option] = result.primary_selection[splitter]
    others = [other for other in node.parameter_space.parameter_options(splitter) if other != option]
    # DFS takes the last one first
    return splitter, [others, [option]]


def split_parameter_space(
    colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask, node: paynt.synthesizer.search_node.SearchNode
) -> list[paynt.synthesizer.search_node.SearchNode]:
    """AR splitting step: pick a parameter to split node's parameter_space on and split its options into subspaces, wrapped as child search nodes.

    :param colored_mdp: anything exposing .parameter_space/.coloring -- any ColoredMdp qualifies
    :param task: the SynthesisTask currently being solved (not read from colored_mdp -- it doesn't carry one)
    :param node: the SearchNode currently being split
    """
    mdp = node.mdp
    assert mdp is not None
    assert not mdp.is_deterministic

    assert node.analysis_result is not None
    results = node.analysis_result.undecided_results()

    # (i) L(h): disagreement among the undecided properties (the objective included) whose OWN scheduler is already fully consistent (a genuine candidate
    # delta_i) -- paper's stated priority
    candidates = [result for result in results if all(len(options) <= 1 for options in result.primary_selection)]
    disagreements = paynt.utils.scoring.compute_incompatibility_levels([result.primary_selection for result in candidates])

    used_options: list[int]
    if disagreements:
        splitter = paynt.utils.scoring.parameters_with_max_incompatibility(disagreements)[0]
        used_options = disagreements[splitter]
    elif scores := scheduler_scores_combined(colored_mdp, task, mdp, results):
        # (ii) v(h) aggregated across every still-relevant property
        splitter = paynt.utils.scoring.parameters_with_max_score(scores)[0]
        # order-preserving union across contributing properties -- NOT sorted(set(...)): suboptions_enumerate
        # builds one singleton bucket per entry in the given order, so reordering would silently change DFS
        # exploration order for multi-constraint specs and, more importantly, must reproduce
        # result.primary_selection[splitter] verbatim when there's exactly one contributing property (N=1)
        used_options = []
        for result in results:
            if len(result.primary_selection[splitter]) > 1:
                for option in result.primary_selection[splitter]:
                    if option not in used_options:
                        used_options.append(option)
    else:
        # (iii) every scheduler left is one design, which does not settle the node (it violates a constraint, or its value is not valid): set it apart
        return node.split(*set_design_apart(colored_mdp, node, candidates[0]))

    if len(used_options) > 1:
        core_suboptions, other_suboptions = node.parameter_space.suboptions_enumerate(splitter, used_options)
    else:
        assert node.parameter_space.parameter_num_options(splitter) > 1
        core_suboptions = node.parameter_space.suboptions_half(splitter)
        other_suboptions = []

    if len(other_suboptions) == 0:
        suboptions = core_suboptions
    else:
        suboptions = [other_suboptions] + core_suboptions  # DFS solves core first

    # construct corresponding child search nodes (SearchNode.split snapshots node's ParentInfo-relevant
    # state and hands it to each freshly-split child, same as add_parent_info used to do manually here)
    return node.split(splitter, suboptions)


class SynthesizerAR(paynt.synthesizer.synthesizer.Synthesizer):
    @property
    def method_name(self) -> str:
        return "AR"

    def check_specification(self, node: paynt.synthesizer.search_node.SearchNode) -> None:
        """Check specification for mdp or smg based on self.colored_mdp."""
        mdp = node.mdp
        assert mdp is not None

        model: paynt.model.model.Mdp
        if self.colored_mdp.feature_kind == "posmg":
            # local, not `import paynt.pomdp.posmg._utils`: that form binds the name `paynt` itself in this
            # function's local scope (Python's static scoping applies to the whole function body regardless
            # of control flow), which would break every other bare `paynt.x` reference below whenever this
            # branch isn't taken
            from paynt.pomdp.posmg import _utils as posmg_utils

            model = posmg_utils.create_smg_from_mdp(self.colored_mdp.feature_info, mdp)
        else:
            model = mdp

        spec = self.task.specification
        if node.constraint_indices is None:
            node.constraint_indices = list(spec.all_constraint_indices())
        # the best and the worst case over several designs bound each of them; a single design is checked like its own model, by the best case alone (see
        # Mdp.check_specification)
        several_designs = node.parameter_space.size > 1
        # a design of the node that satisfies the whole specification, found among the designs the best cases come from, and its value of the objective
        admissible_assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
        admissible_value: Any = None

        results: list[paynt.specification.property_result.MdpPropertyResult | None] = [None for _ in spec.constraints]
        for index in node.constraint_indices:
            constraint = spec.constraints[index]
            result = paynt.specification.property_result.MdpPropertyResult(constraint)
            results[index] = result

            # the best case: if it violates the constraint, every design does
            result.primary = model.model_check_property(constraint)
            if not result.primary.sat and not (several_designs and constraint.best_case_is_unknown(result.primary.value)):
                result.sat = False
                break
            if not several_designs:
                result.sat = True
                continue

            # a design the best case comes from may satisfy the whole specification, which says nothing about the other designs
            result.primary_selection, consistent = self.colored_mdp.scheduler_is_consistent(mdp, node, result.primary.result, spec)
            if consistent:
                assignment = node.parameter_space.assume_options_copy(result.primary_selection)
                accepting, value = self.colored_mdp.build_assignment(assignment).check_specification(spec).accepting_dtmc(spec)
                better = admissible_assignment is None or (spec.optimality is not None and value is not None and spec.optimality.op(value, admissible_value))
                if accepting and better:
                    admissible_assignment, admissible_value = assignment, value

            # the worst case: if it satisfies the constraint, every design does
            result.secondary = model.model_check_property(constraint, alt=True)
            if mdp.is_deterministic and result.primary.value != result.secondary.value:
                logger.warning("WARNING: model is deterministic but min<max")
            if not constraint.best_case_is_unknown(result.primary.value):
                if result.secondary.sat:
                    result.sat = True
            elif result.secondary.value == math.inf:
                # the least reward is infinite too: no design surely reaches the target
                result.sat = False
                break
            # else some design may not surely reach the target, which no least reward satisfying the constraint rules out

        spec_result = paynt.specification.property_result.MdpSpecificationResult()
        spec_result.constraints_result = paynt.specification.property_result.ConstraintsResult(results)

        if spec.optimality is not None and spec_result.constraints_result.sat is not False:
            spec_result.optimality_result = self.check_objective(node, model, several_designs)

        spec_result.evaluate(node.parameter_space, admissible_assignment, admissible_value)
        node.analysis_result = spec_result

    def check_objective(
        self, node: paynt.synthesizer.search_node.SearchNode, model: paynt.model.model.Mdp, several_designs: bool
    ) -> paynt.specification.property_result.MdpOptimalityResult:
        mdp = node.mdp
        assert mdp is not None
        spec = self.task.specification
        opt = spec.optimality
        assert opt is not None
        result = paynt.specification.property_result.MdpOptimalityResult(opt)

        # the best case: no design does better
        result.primary = model.model_check_property(opt)
        if several_designs and opt.best_case_is_unknown(result.primary.value):
            # some design may improve the optimum, unless the least reward is infinite too: then no design surely reaches the target
            improves = model.model_check_property(opt, alt=True).value != math.inf
        else:
            improves = result.primary.improves_optimum
        if not improves:
            result.can_improve = False
            return result

        result.can_improve = True
        if several_designs:
            result.primary_selection, consistent = self.colored_mdp.scheduler_is_consistent(mdp, node, result.primary.result, spec)
            if not consistent:
                return result
            assignment = node.parameter_space.assume_options_copy(result.primary_selection)
        else:
            assignment = node.parameter_space
        # the design the best case comes from, the best design of the node
        res = self.colored_mdp.build_assignment(assignment).check_specification(spec)
        assert res.constraints_result is not None
        value = res.optimality_result.value
        if res.constraints_result.sat and opt.improves_optimum(value):
            result.improving_assignment = assignment
            result.improving_value = value
            result.can_improve = False
        elif several_designs and not opt.result_valid(value):
            # it never surely reaches the target, the other designs may: the best case is infinite, or Storm's scheduler for a finite one is not the best (for
            # a minimized reward it can pick a choice that leaves the target unreachable)
            pass
        elif not opt.improves_optimum(value) or not several_designs or mdp.is_deterministic:
            # it does not improve the optimum, so no design does; or the designs all behave like it
            result.can_improve = False
        # else it improves the optimum but violates a constraint, which other designs may satisfy
        return result

    def verify_parameter_space(self, node: paynt.synthesizer.search_node.SearchNode) -> None:
        parent_selected_choices = node.parent_info.selected_choices if node.parent_info is not None else None
        node.mdp, node.selected_choices = self.colored_mdp.build(node.parameter_space, parent_selected_choices)
        assert self.stat is not None

        # TODO include iteration_game in iteration? is it necessary?
        if self.colored_mdp.feature_kind == "posmg":
            self.stat.iteration_game(node.mdp.states)
        else:
            self.stat.iteration(node.mdp)

        self.check_specification(node)

    def update_optimum(self, node: paynt.synthesizer.search_node.SearchNode) -> None:
        assert node.analysis_result is not None
        ia = node.analysis_result.improving_assignment
        if ia is None:
            return
        optimality = self.task.specification.optimality
        if optimality is None:
            self.best_assignment = ia
            return
        iv = node.analysis_result.improving_value
        if not optimality.improves_optimum(iv):
            return
        optimality.update_optimum(iv)
        self.best_assignment = ia
        self.best_assignment_value = iv
        # logger.info(f"value {round(iv,4)} achieved after {round(paynt.utils.timer.GlobalTimer.read(),2)} seconds")
        if self.colored_mdp.feature_kind == "pomdp":
            from paynt.pomdp import _utils as pomdp_utils

            assert self.stat is not None
            self.stat.new_fsc_found(node.analysis_result.improving_value, ia, pomdp_utils.policy_size(self.colored_mdp, ia))

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        nodes = [node]
        while nodes:
            if self.resource_limit_reached():
                break
            node = nodes.pop(-1)
            self.verify_parameter_space(node)
            self.update_optimum(node)
            if not self.task.specification.has_optimality and self.best_assignment is not None:
                break
            # break
            assert node.analysis_result is not None
            if node.analysis_result.can_improve is False:
                self.explore(node.parameter_space)
                continue
            # undecided
            child_nodes = self.split_undecided_space(node)
            nodes = nodes + child_nodes
        return self.best_assignment

    def split_undecided_space(self, node: paynt.synthesizer.search_node.SearchNode) -> list[paynt.synthesizer.search_node.SearchNode]:
        """Overridable hook: the default just delegates to the shared split_parameter_space free function."""
        return split_parameter_space(self.colored_mdp, self.task, node)
