from __future__ import annotations


import paynt.colored_mdp
import paynt.task
import paynt.synthesizer.search_node
import paynt.parameter_space.parameter_space
import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_ar
import paynt.synthesizer.synthesizer_cegis

import paynt.parameter_space.smt
import paynt.utils.timer

import logging

logger = logging.getLogger(__name__)


class StageControl:
    """AR-CEGIS adaptivity: switch between ar/cegis, allocate more time to the more efficient method."""

    # whether only AR is performed
    only_ar = False
    # whether 1 AR followed by only CEGIS is performed
    only_cegis = False
    # whether adaptive hybrid is enabled
    adaptive_hybrid = True

    def __init__(self, parameter_space_size: int):
        # timings
        self.timer_ar = paynt.utils.timer.Timer()
        self.timer_cegis = paynt.utils.timer.Timer()

        self.parameter_space_size = parameter_space_size
        self.pruned_ar: float = 0
        self.pruned_cegis: float = 0

        # multiplier to derive time allocated for cegis
        # time_ar * factor = time_cegis
        # =1 is fair, >1 favours cegis, <1 favours ar
        self.cegis_efficiency: float = 1

    def start_ar(self) -> None:
        self.timer_cegis.stop()
        self.timer_ar.start()

    def start_cegis(self) -> None:
        self.timer_ar.stop()
        self.timer_cegis.start()

    def prune_ar(self, pruned: int) -> None:
        self.pruned_ar += pruned / self.parameter_space_size

    def prune_cegis(self, pruned: int) -> None:
        self.pruned_cegis += pruned / self.parameter_space_size

    def cegis_has_time(self) -> bool:
        """:return: True if cegis still has some time."""
        # whether only AR is performed
        if StageControl.only_ar:
            return False

        # whether only CEGIS is performed
        if StageControl.only_cegis:
            return True

        # whether CEGIS has more time
        if self.timer_cegis.read() < self.timer_ar.read() * self.cegis_efficiency:
            return True

        # stop CEGIS
        self.timer_cegis.stop()

        if StageControl.adaptive_hybrid:
            if self.pruned_ar == 0 and self.pruned_cegis == 0:
                self.cegis_efficiency = 1
            elif self.pruned_ar == 0 and self.pruned_cegis > 0:
                self.cegis_efficiency = 2
            elif self.pruned_ar > 0 and self.pruned_cegis == 0:
                self.cegis_efficiency = 0.5
            else:
                success_rate_cegis = self.pruned_cegis / self.timer_cegis.read()
                success_rate_ar = self.pruned_ar / self.timer_ar.read()
                self.cegis_efficiency = success_rate_cegis / success_rate_ar

        return False


class SynthesizerHybrid(paynt.synthesizer.synthesizer_ar.SynthesizerAR, paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS):
    @property
    def method_name(self) -> str:
        return "hybrid"

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:

        self.conflict_generator.initialize()
        # deliberately not threading self.constraint through here, even though it exists on this instance
        # (inherited from SynthesizerCEGIS.__init__ via this class's MRO): Hybrid doesn't support custom
        # constraints yet, so this omission is intentional, not an oversight.
        smt_solver = paynt.parameter_space.smt.SmtSolver(self.colored_mdp.parameter_space)

        # AR-CEGIS loop
        nodes = [node]
        self.stage_control = StageControl(node.parameter_space.size)
        while nodes:
            if self.resource_limit_reached():
                break

            # initiate AR analysis
            self.stage_control.start_ar()

            # choose parameter space
            node = nodes.pop(-1)

            # reset SMT solver level
            smt_solver.level(node.refinement_depth)

            # analyze the parameter space
            self.verify_parameter_space(node)
            self.update_optimum(node)
            assert node.analysis_result is not None
            if not node.analysis_result.can_improve:
                self.explore(node.parameter_space)
                self.stage_control.prune_ar(node.parameter_space.size)
                continue

            # undecided: initiate CEGIS analysis
            self.stage_control.start_cegis()

            # construct priority parameter subspace that corresponds to primary scheduler
            if node.analysis_result.optimality_result is not None:
                result = node.analysis_result.optimality_result
            else:
                cr = node.analysis_result.constraints_result
                assert cr is not None
                # index 0 is wrong once a spec has 2+ constraints: constraint 0 may already be sat (dropped
                # by an ancestor node, see SearchNode.collect_parent_info), leaving results[0] None
                result = cr.results[cr.undecided_constraints[0]]
            priority_node = self.search_node_type(node.parameter_space.assume_options_copy(result.primary_selection))

            # explore parameter space assignments
            parameter_space_explored = False
            while True:
                if not self.stage_control.cegis_has_time() or self.resource_limit_reached():
                    break  # CEGIS timeout (or a global resource limit was reached)

                node.encode(smt_solver)
                # assignment = smt_solver.pick_assignment(node)
                assignment = smt_solver.pick_assignment_priority(node, priority_node)
                if assignment is None:
                    parameter_space_explored = True
                    break  # explored whole parameter space

                conflicts, accepting_assignment = self.analyze_parameter_space_assignment_cegis(node, assignment)
                pruned = smt_solver.exclude_conflicts(node, assignment, conflicts)
                assert self.explored is not None
                self.explored += pruned
                self.stage_control.prune_cegis(pruned)

                if accepting_assignment is not None:
                    self.best_assignment = accepting_assignment
                    if not self.task.specification.can_be_improved():
                        return self.best_assignment

                # assignment is UNSAT: move on to the next assignment

            if parameter_space_explored:
                continue

            child_nodes = self.split_undecided_space(node)
            nodes = nodes + child_nodes

        return self.best_assignment
