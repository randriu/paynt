from __future__ import annotations

from typing import Any, cast

import paynt.colored_mdp
import paynt.synthesizer.synthesizer
import paynt.parameter_space.parameter_space
import paynt.task
import paynt.specification.property
import paynt.model.model
import paynt.utils.error_handling
import paynt.utils.scoring
import paynt.mdp_family._utils
import paynt.mdp_family.result
from paynt.mdp_family.policy_tree import Policy, PolicyTree, PolicyTreeNode

import logging

logger = logging.getLogger(__name__)


class MdpFamilyResult:
    def __init__(self) -> None:
        # if None, then family is undediced
        # if False, then all family members are UNSAT
        # otherwise, contains a satisfying policy for all MDPs in the family
        self.policy: list[int | None] | bool | None = None

        self.game_policy: list[int | None] | None = None
        self.parameter_selection: list[list[int]] | None = None
        self.splitter: int | None = None


class PolicyTreeSynthesizer(paynt.synthesizer.synthesizer.Synthesizer):
    # if True, tree leaves will be double-checked after synthesis
    double_check_policy_tree_leaves = False

    # set by evaluate_all(), read by run()/export_evaluation_result()
    policy_tree: PolicyTree

    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask):
        paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, "the policy tree")
        super().__init__(colored_mdp, task)

    @property
    def method_name(self) -> str:
        return "AR (policy tree)"

    def verify_policy(self, selected_choices: Any, prop: paynt.specification.property.Property, policy: list[int | None]) -> bool:
        """:param selected_choices: the compatible-choices bitmask to verify against -- callers outside this class have no PolicyTreeNode to read it from, so it
        must be supplied explicitly (see ParameterSpaceEvaluation.selected_choices, captured at decision time)"""
        _, mdp = paynt.mdp_family._utils.fix_and_apply_policy_to_parameter_space(self.colored_mdp, selected_choices, policy)
        policy_result = mdp.model_check_property(prop, alt=True)
        assert self.stat is not None
        self.stat.iteration(mdp)
        return policy_result.sat

    def solve_singleton(self, node: PolicyTreeNode, prop: paynt.specification.property.Property) -> list[int | None] | bool:
        assert node.mdp is not None
        assert self.stat is not None
        result = node.mdp.model_check_property(prop)
        self.stat.iteration(node.mdp)
        if not result.sat:
            return False
        assert result.result.scheduler is not None
        return paynt.mdp_family._utils.scheduler_to_policy(self.colored_mdp, result.result.scheduler, node.mdp)

        # uncomment below to preemptively double-check the policy
        # paynt.mdp_family.policy_tree.double_check_policy(self.colored_mdp, node, prop, policy)

    def solve_game_abstraction(self, node: PolicyTreeNode, prop: paynt.specification.property.Property, game_solver: Any) -> tuple[list[int | None], bool]:
        # construct and solve the game abstraction
        # logger.debug("solving game abstraction...")
        assert node.mdp is not None
        assert self.stat is not None

        game_solver.solve_sg(node.selected_choices)
        # game_solver.solve_smg(node.selected_choices)

        game_value = game_solver.solution_value
        self.stat.iteration_game(node.mdp.states)
        game_sat = prop.satisfies_threshold_within_precision(game_value)
        # logger.debug("game solved, value is {}".format(game_value))
        game_policy = game_solver.solution_state_to_player1_action
        # fix irrelevant choices
        game_policy_fixed = paynt.mdp_family._utils.empty_policy(self.colored_mdp)
        info = cast(paynt.mdp_family._utils.MdpFamilyInfo, self.colored_mdp.feature_info)
        for state, action in enumerate(game_policy):
            if action < info.num_actions:
                game_policy_fixed[state] = action
        game_policy = game_policy_fixed
        return game_policy, game_sat

    def state_to_choice_to_parameter_selection(self, state_to_choice: list[int | None]) -> tuple[Any, list[list[int]]]:
        if self.task.discard_unreachable_choices:
            state_to_choice = paynt.model.model.ModelIndex.discard_unreachable_choices(
                self.colored_mdp.underlying_mdp, self.colored_mdp.choice_destinations, state_to_choice
            )
        scheduler_choices = paynt.model.model.ModelIndex.state_to_choice_to_choices(self.colored_mdp.underlying_mdp, state_to_choice)
        parameter_selection = self.colored_mdp.coloring.collectHoleOptions(scheduler_choices)
        return scheduler_choices, parameter_selection

    def parse_game_scheduler(self, game_solver: Any) -> tuple[Any, list[list[int]], list[float]]:
        state_to_choice = game_solver.solution_state_to_quotient_choice.copy()
        scheduler_choices, parameter_selection = self.state_to_choice_to_parameter_selection(state_to_choice)
        state_values = game_solver.solution_state_values
        return scheduler_choices, parameter_selection, state_values

    def verify_parameter_space(self, node: PolicyTreeNode, game_solver: Any, prop: paynt.specification.property.Property) -> MdpFamilyResult:
        # logger.info("investigating parameter space of size {}".format(node.parameter_space.size))
        parent_selected_choices = node.parent_info.selected_choices if node.parent_info is not None else None
        node.mdp, node.selected_choices = self.colored_mdp.build(node.parameter_space, parent_selected_choices)
        mdp_family_result = MdpFamilyResult()

        if node.parameter_space.size == 1:
            mdp_family_result.policy = self.solve_singleton(node, prop)
            return mdp_family_result

        if node.candidate_policy is None:
            game_policy, game_sat = self.solve_game_abstraction(node, prop, game_solver)
        else:
            game_policy = node.candidate_policy
            game_sat = False

        mdp_family_result.game_policy = game_policy
        if game_sat:
            mdp_family_result.policy = game_policy
            return mdp_family_result

        # solve primary direction for the MDP abstraction
        assert node.mdp is not None
        assert self.stat is not None
        mdp_result = node.mdp.model_check_property(prop)
        self.stat.iteration(node.mdp)
        # logger.debug("primary-primary direction solved, value is {}".format(mdp_value))
        if not mdp_result.sat:
            mdp_family_result.policy = False
            return mdp_family_result

        # undecided: choose scheduler choices to be used for splitting
        scheduler_choices, parameter_selection, state_values = self.parse_game_scheduler(game_solver)

        splitter = self.choose_splitter(node.parameter_space, prop, scheduler_choices, state_values, parameter_selection)
        mdp_family_result.splitter = splitter
        mdp_family_result.parameter_selection = parameter_selection
        return mdp_family_result

    def choose_splitter(
        self,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        prop: paynt.specification.property.Property,
        scheduler_choices: Any,
        state_values: list[float],
        parameter_selection: list[list[int]],
    ) -> int:
        inconsistent_assignments = {parameter: options for parameter, options in enumerate(parameter_selection) if len(options) > 1}
        if len(inconsistent_assignments) == 0:
            # pick any parameter with multiple options involved in the parameter selection
            for parameter, options in enumerate(parameter_selection):
                if parameter_space.parameter_num_options(parameter) > 1 and len(options) > 0:
                    return parameter
            # pick any parameter with multiple options
            # logger.debug("picking an arbitrary parameter...")
            for parameter in range(parameter_space.num_parameters):
                if parameter_space.parameter_num_options(parameter) > 1:
                    return parameter
        if len(inconsistent_assignments) == 1:
            for parameter in inconsistent_assignments.keys():
                return parameter

        # compute scores for inconsistent parameters
        scores = self.compute_scores(prop, scheduler_choices, state_values, inconsistent_assignments)
        splitters = paynt.utils.scoring.parameters_with_max_score(scores)
        return splitters[0]

    def compute_scores(
        self, prop: paynt.specification.property.Property, scheduler_choices: Any, state_values: list[float], inconsistent_assignments: dict[int, list[int]]
    ) -> dict[int, float]:
        mdp = self.colored_mdp.underlying_mdp
        choice_values = paynt.model.model.ModelIndex.choice_values(mdp, prop, state_values)
        expected_visits = paynt.model.model.ModelIndex.compute_expected_visits(
            mdp, prop, scheduler_choices, disable_expected_visits=self.task.disable_expected_visits
        )
        underlying_mdp_choice_map = list(range(self.colored_mdp.underlying_mdp.nr_choices))
        return paynt.utils.scoring.estimate_scheduler_difference(
            self.colored_mdp, self.colored_mdp.underlying_mdp, underlying_mdp_choice_map, inconsistent_assignments, choice_values, expected_visits
        )

    def assign_candidate_policy(
        self,
        parameter_subspaces: list[paynt.parameter_space.parameter_space.ParameterSpace],
        candidate_policies: list[list[int | None] | None],
        parameter_selection: list[list[int]],
        splitter: int,
        policy: list[int | None] | None,
    ) -> None:
        """Fill in candidate_policies[i] = policy for whichever parameter_subspaces[i] contains the parameter
        selection, in place -- parameter_subspaces are plain ParameterSpace values here (not yet wrapped as
        PolicyTreeNode children), so the candidate policy can't be attached directly until attach_children
        wraps them (see evaluate_all)."""
        policy_consistent = all(len(options) <= 1 for options in parameter_selection)
        if not policy_consistent:
            return
        # associate the branch of the split that contains the parameter selection with the policy
        used_options = parameter_selection[splitter]
        if len(used_options) != 1:
            # not sure what to do in this case
            return
        option = used_options[0]
        for index, parameter_subspace in enumerate(parameter_subspaces):
            if option in parameter_subspace.parameter_options(splitter):
                candidate_policies[index] = policy
                return

    def split(
        self,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        prop: paynt.specification.property.Property,
        parameter_selection: list[list[int]],
        splitter: int,
        policy: list[int | None] | None,
    ) -> tuple[list[list[int]], list[paynt.parameter_space.parameter_space.ParameterSpace], list[list[int | None] | None]]:
        # split the parameter
        used_options = parameter_selection[splitter]
        if len(used_options) > 1:
            # used_options = used_options[0:1]
            core_suboptions = [[option] for option in used_options]
            other_suboptions_list = [option for option in parameter_space.parameter_options(splitter) if option not in used_options]
            suboptions: list[list[int]]
            if other_suboptions_list:
                suboptions = [other_suboptions_list] + core_suboptions  # DFS solves core first
            else:
                suboptions = core_suboptions
        else:
            options = parameter_space.parameter_options(splitter)
            assert len(options) > 1
            half = len(options) // 2
            suboptions = [options[:half], options[half:]]

        parameter_subspaces = parameter_space.split(splitter, suboptions)
        candidate_policies: list[list[int | None] | None] = [None for _ in parameter_subspaces]

        if not self.task.discard_unreachable_choices:
            self.assign_candidate_policy(parameter_subspaces, candidate_policies, parameter_selection, splitter, policy)

        return suboptions, parameter_subspaces, candidate_policies

    def evaluate_all(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, prop: paynt.specification.property.Property, keep_value_only: bool = False
    ) -> list[paynt.synthesizer.synthesizer.ParameterSpaceEvaluation]:
        assert not prop.reward, "expecting reachability probability propery"
        assert self.stat is not None
        game_solver = paynt.mdp_family._utils.build_game_abstraction_solver(self.colored_mdp, prop)
        policy_tree = PolicyTree(parameter_space)

        undecided_leaves = [policy_tree.root]
        while undecided_leaves:
            if self.resource_limit_reached():
                break

            # gi = self.stat.iterations_game
            # if gi is not None and gi > 1000:
            #     return None

            node = undecided_leaves.pop(-1)
            result = self.verify_parameter_space(node, game_solver, prop)
            node.candidate_policy = None

            if result.policy is not None:
                self.explore(node.parameter_space)
                if node != policy_tree.root:
                    node.mdp = None
                if result.policy is False:
                    node.sat = False
                else:
                    assert result.policy is not True
                    node.sat = True
                    node.policy_index = policy_tree.new_policy(result.policy)
                continue

            # refine
            assert result.parameter_selection is not None and result.splitter is not None
            suboptions, parameter_subspaces, candidate_policies = self.split(
                node.parameter_space, prop, result.parameter_selection, result.splitter, result.game_policy
            )
            if node != policy_tree.root:
                node.mdp = None
            node.attach_children(result.splitter, suboptions, parameter_subspaces, candidate_policies)
            undecided_leaves += node.child_nodes

        if PolicyTreeSynthesizer.double_check_policy_tree_leaves:
            policy_tree.double_check(self.colored_mdp, prop)
        policy_tree.print_stats()

        self.stat.num_mdps_total = self.colored_mdp.parameter_space.size
        self.stat.num_mdps_sat = sum([n.parameter_space.size for n in policy_tree.collect_sat()])
        self.stat.num_nodes = len(policy_tree.collect_all())
        self.stat.num_leaves = len(policy_tree.collect_leaves())
        self.stat.num_policies = len(policy_tree.policies)
        postprocessing_time = policy_tree.postprocess(self.colored_mdp, prop)
        policy_tree.print_stats()
        self.stat.postprocessing_time = postprocessing_time
        self.stat.num_nodes_merged = len(policy_tree.collect_all())
        self.stat.num_leaves_merged = len(policy_tree.collect_leaves())
        self.stat.num_policies_merged = len(policy_tree.policies)
        self.policy_tree = policy_tree

        # convert policy tree to parameter space evaluations
        evaluations = []
        for node in policy_tree.collect_leaves():
            policy: Policy | None
            if node.sat:
                assert node.policy_index is not None
                policy = policy_tree.policies[node.policy_index]
            else:
                policy = None
            evaluation = paynt.synthesizer.synthesizer.ParameterSpaceEvaluation(
                node.parameter_space, None, node.sat, policy=policy, selected_choices=node.selected_choices
            )
            evaluations.append(evaluation)
        return evaluations

    def run(self, optimum_threshold: Any = None) -> paynt.mdp_family.result.PolicyTreeResult:
        evaluations = self.evaluate()
        success = len(evaluations) > 0 and all(evaluation.sat for evaluation in evaluations)
        return paynt.mdp_family.result.PolicyTreeResult(success, policy_tree=self.policy_tree)

    def export_evaluation_result(self, evaluations: list[Any], export_filename_base: str) -> None:
        import json

        policies = self.policy_tree.extract_policies(self.colored_mdp)
        policies_json = {}
        for _index, key_value in enumerate(policies.items()):
            policy_id, policy = key_value
            policy_json = paynt.mdp_family._utils.policy_to_json(policy)
            policies_json[policy_id] = policy_json
        policies_string = json.dumps(policies_json, indent=4)

        policies_filename = export_filename_base + ".json"
        with open(policies_filename, "w") as file:
            file.write(policies_string)

        logger.info(f"exported policies to {policies_filename}")

        tree = self.policy_tree.extract_policy_tree(self.colored_mdp)
        tree_filename = export_filename_base + ".dot"
        with open(tree_filename, "w") as file:
            file.write(tree.source)
        logger.info(f"exported policy tree to {tree_filename}")

        tree_visualization_filename = export_filename_base + ".png"
        tree.render(export_filename_base, format="png", cleanup=True)  # using export_filename_base since graphviz appends .png by default
        logger.info(f"exported policy tree visualization to {tree_visualization_filename}")
