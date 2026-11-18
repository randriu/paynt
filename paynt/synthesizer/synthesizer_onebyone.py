from __future__ import annotations

from typing import Any

import paynt.parameter_space.parameter_space
import paynt.synthesizer.search_node
import paynt.synthesizer.synthesizer
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)


class SynthesizerOneByOne(paynt.synthesizer.synthesizer.Synthesizer):
    @property
    def method_name(self) -> str:
        return "1-by-1"

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:

        for parameter_combination in node.parameter_space.all_combinations():
            if self.resource_limit_reached():
                break

            assignment = node.parameter_space.construct_assignment(parameter_combination)
            try:
                dtmc = self.colored_mdp.build_assignment(assignment)
            except paynt.utils.error_handling.DeadlockedDesigns as error:
                logger.warning(error)
                self.explore(assignment)
                continue
            assert self.stat is not None
            self.stat.iteration(dtmc)
            result = dtmc.check_specification(self.task.specification, short_evaluation=True)
            self.explore(assignment)

            accepting, improving_value = result.accepting_dtmc(self.task.specification)
            if accepting:
                self.best_assignment = assignment
            if improving_value is not None:
                assert self.task.specification.optimality is not None
                self.task.specification.optimality.update_optimum(improving_value)
                self.best_assignment_value = improving_value
            if accepting and not self.task.specification.can_be_improved():
                return self.best_assignment

        return self.best_assignment

    def evaluate_all(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, prop: Any, keep_value_only: bool = False) -> list[Any]:

        evaluations = []
        assert self.stat is not None
        for parameter_combination in parameter_space.all_combinations():
            if self.resource_limit_reached():
                break
            assignment = parameter_space.construct_assignment(parameter_combination)
            try:
                model = self.colored_mdp.build_assignment(assignment)
            except paynt.utils.error_handling.DeadlockedDesigns as error:
                # not a design: it has no evaluation
                logger.warning(error)
                self.explore(assignment)
                continue
            self.stat.iteration(model)
            result = model.model_check_property(prop)
            if keep_value_only:
                evaluation: Any = result.value
            else:
                # a member of a family is an MDP, so it comes with the policy that attains its value; a DTMC needs none
                policy = None
                if result.sat and self.colored_mdp.feature_kind == "family":
                    from paynt.mdp_family import _utils as mdp_family_utils

                    actions = mdp_family_utils.scheduler_to_policy(self.colored_mdp, result.result.get_scheduler(), model)
                    policy = (actions, [state for state, action in enumerate(actions) if action is not None])
                evaluation = paynt.synthesizer.synthesizer.ParameterSpaceEvaluation(assignment, result.value, result.sat, policy)
            evaluations.append(evaluation)
            self.explore(assignment)
        return evaluations

    def export_evaluation_result(self, evaluations: list[Any], export_filename_base: str) -> None:
        import json

        parameter_space_to_evaluation_parsed = []
        for evaluation in evaluations:
            parameter_space = evaluation.parameter_space
            policy = evaluation.policy
            if not evaluation.sat:
                policy = "UNSAT"
            elif policy is None:
                policy = "SAT"
            else:
                from paynt.mdp_family import _utils as mdp_family_utils

                policy = mdp_family_utils.policy_to_state_valuation_actions(self.colored_mdp, policy)
            parameter_space_to_evaluation_parsed.append((str(parameter_space), policy))
        policies_string = json.dumps(parameter_space_to_evaluation_parsed, indent=2)
        policies_filename = export_filename_base + ".json"
        with open(policies_filename, "w") as file:
            file.write(policies_string)
        logger.info(f"exported satisfied members and correponding policies to {policies_filename}")
