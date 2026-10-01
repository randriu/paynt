from __future__ import annotations

from typing import Any

import paynt.colored_mdp
import paynt.task
import paynt.synthesizer.search_node
import paynt.synthesizer.synthesizer
import paynt.synthesizer.conflict_generator.auto
import paynt.synthesizer.conflict_generator.dtmc
import paynt.synthesizer.conflict_generator.mdp
import paynt.parameter_space.parameter_space
import paynt.parameter_space.smt
import paynt.parameter_space.constraints

import logging

logger = logging.getLogger(__name__)


class SynthesizerCEGIS(paynt.synthesizer.synthesizer.Synthesizer):
    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask):
        super().__init__(colored_mdp, task)

        self.conflict_generator = self.choose_conflict_generator(colored_mdp, task)

        # assert that no reward formula is maximizing
        assert not self.task.specification.contains_maximizing_reward_properties, (
            "Cannot use CEGIS for maximizing reward formulae -- consider using AR or hybrid methods."
        )

        # unlike SynthesizerSMPMC, CEGIS's pre-existing behavior (no constraint at all) is the default --
        # a constraint is only built when the user explicitly asks for one via --constraint
        self.constraint = paynt.parameter_space.constraints.build_constraint(task.constraint_name) if task.constraint_name else None

    def choose_conflict_generator(
        self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask
    ) -> paynt.synthesizer.conflict_generator.dtmc.ConflictGeneratorDtmc:
        if task.conflict_generator_type == "mdp":
            conflict_generator: paynt.synthesizer.conflict_generator.dtmc.ConflictGeneratorDtmc = paynt.synthesizer.conflict_generator.mdp.ConflictGeneratorMdp(
                colored_mdp, task
            )
        else:
            # default conflict generator: the DTMC one, unless the model of an assignment still has nondeterminism
            conflict_generator = paynt.synthesizer.conflict_generator.auto.ConflictGeneratorAuto(colored_mdp, task)
        return conflict_generator

    @property
    def method_name(self) -> str:
        return "CEGIS " + self.conflict_generator.name

    def collect_conflict_requests(self, node: paynt.synthesizer.search_node.SearchNode, mc_result: Any) -> list[tuple[int, Any, Any]]:
        """Construct conflict request wrt each unsatisfiable property, pack such properties as well as their MDP results (if available)"""
        conflict_requests: list[tuple[int, Any, Any]] = []
        assert node.constraint_indices is not None
        for index in node.constraint_indices:
            member_result = mc_result.constraints_result.results[index]
            # short_evaluation=True (see analyze_parameter_space_assignment_cegis) stops checking constraints
            # as soon as one is UNSAT, leaving later indices None -- a spec with 2+ constraints reaches this
            # routinely, not just as an edge case
            if member_result is None or member_result.sat:
                continue
            prop: Any = self.task.specification.constraints[index]
            parameter_space_result = None
            if node.analysis_result is not None:
                assert node.analysis_result.constraints_result is not None
                parameter_space_result = node.analysis_result.constraints_result.results[index]
            conflict_requests.append((index, prop, parameter_space_result))
        if self.task.specification.optimality is not None:
            member_result = mc_result.optimality_result
            index = len(self.task.specification.constraints)
            prop = self.task.specification.optimality
            parameter_space_result = node.analysis_result.optimality_result if node.analysis_result is not None else None
            conflict_requests.append((index, prop, parameter_space_result))

        return conflict_requests

    def analyze_parameter_space_assignment_cegis(
        self, node: paynt.synthesizer.search_node.SearchNode, assignment: paynt.parameter_space.parameter_space.ParameterSpace
    ) -> tuple[list[Any], paynt.parameter_space.parameter_space.ParameterSpace | None]:
        """Analyze a single fixed assignment via CEGIS conflict analysis.

        :return: (1) list of conflicts to exclude from design space (might be empty)
        :return: (2) accepting assignment (or None)
        """
        assert node.mdp is not None, "analyzed parameter space does not have an associated underlying MDP"
        assert self.stat is not None

        dtmc = self.colored_mdp.build_assignment(assignment)
        self.stat.iteration(dtmc)
        result = dtmc.check_specification(self.task.specification, node.constraint_indices, short_evaluation=True)
        # analyze model checking results
        accepting_assignment = None
        accepting, improving_value = result.accepting_dtmc(self.task.specification)
        if accepting:
            accepting_assignment = assignment
        if improving_value is not None:
            assert self.task.specification.optimality is not None
            self.task.specification.optimality.update_optimum(improving_value)
            self.best_assignment_value = improving_value
        if accepting and not self.task.specification.can_be_improved():
            return [], accepting_assignment

        conflict_requests = self.collect_conflict_requests(node, result)
        conflicts = self.conflict_generator.construct_conflicts(node, assignment, dtmc, conflict_requests)

        return conflicts, accepting_assignment

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:

        # build the induced sub-MDP, mapping mdp states to parameter indices
        parent_selected_choices = node.parent_info.selected_choices if node.parent_info is not None else None
        node.mdp, node.selected_choices = self.colored_mdp.build(node.parameter_space, parent_selected_choices)
        self.conflict_generator.initialize()

        # use sketch design space as a SAT baseline (TODO why?)
        smt_solver = paynt.parameter_space.smt.SmtSolver(self.colored_mdp.parameter_space, self.constraint, self.colored_mdp, self.task)

        # CEGIS loop
        assignment = smt_solver.pick_assignment(node)
        while assignment is not None:
            if self.resource_limit_reached():
                break

            conflicts, accepting_assignment = self.analyze_parameter_space_assignment_cegis(node, assignment)
            if accepting_assignment is not None:
                self.best_assignment = accepting_assignment
                if not self.task.specification.can_be_improved():
                    return self.best_assignment

            pruned = smt_solver.exclude_conflicts(node, assignment, conflicts)
            assert self.explored is not None
            self.explored += pruned

            # construct next assignment
            assignment = smt_solver.pick_assignment(node)
        return self.best_assignment
