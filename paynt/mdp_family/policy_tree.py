"""Representation of a policy tree: a tree over parameter subspaces of a family, where each leaf is either unsatisfiable or associated with a policy that
satisfies every member of its subspace."""

from __future__ import annotations

from typing import Any

import paynt.colored_mdp
import paynt.synthesizer.search_node
import paynt.parameter_space.parameter_space
import paynt.mdp_family._utils
import paynt.utils.error_handling
from paynt.specification.property import Property
import paynt.utils.timer

import logging

logger = logging.getLogger(__name__)

# disable logging when importing graphviz to suppress warnings
logging.disable(logging.CRITICAL)
import graphviz  # noqa: E402

logging.disable(logging.NOTSET)

# a policy over the underlying MDP: for each state, the executed action (or None if unconstrained/unreachable),
# plus a mask of the state indices where the action is actually defined
Policy = tuple[list[int | None], list[int]]


def policies_are_compatible(policy1: Policy, policy2: Policy) -> bool:
    policy1_actions, policy1_mask = policy1
    policy2_actions, _ = policy2
    for state in policy1_mask:
        a1 = policy1_actions[state]
        a2 = policy2_actions[state]
        if a2 is not None and a1 != a2:
            return False
    return True


def merge_policies(policy1: Policy, policy2: Policy) -> Policy | None:
    """Attempt to merge multiple policies into one.

    :returns: one policy or None if some policies were incompatible
    """
    if not policies_are_compatible(policy1, policy2):
        return None
    policy1_actions, _ = policy1
    policy2_actions, _ = policy2
    policy = [a1 if a1 is not None else policy2_actions[state] for state, a1 in enumerate(policy1_actions)]
    mask = [state for state, action in enumerate(policy) if action is not None]
    return (policy, mask)


def merge_policies_exclusively(policy1: Policy, policy2: Policy) -> tuple[list[int | None], list[int | None]]:
    policy1_actions, _ = policy1
    policy2_actions, _ = policy2
    policy12 = policy1_actions.copy()
    policy21 = policy2_actions.copy()
    for state, a1 in enumerate(policy1_actions):
        a2 = policy2_actions[state]
        if a1 is None:
            policy12[state] = a2
        if a2 is None:
            policy21[state] = a1
    return policy12, policy21


def double_check_policy(
    colored_mdp: paynt.colored_mdp.ColoredMdp, node: paynt.synthesizer.search_node.SearchNode, prop: Property, policy: list[int | None]
) -> None:
    """Re-verify (at tighter precision) that policy is actually SAT for node's parameter space -- used by PolicyTreeNode.double_check, itself only run when
    PolicyTreeSynthesizer.double_check_policy_tree_leaves is enabled.

    Takes colored_mdp/node explicitly (not a Synthesizer instance) since it needs no search state, just the representation and a policy to check.
    """
    (policy_fixed, _), mdp = paynt.mdp_family._utils.fix_and_apply_policy_to_parameter_space(colored_mdp, node.selected_choices, policy)
    if paynt.mdp_family._utils.policy_gets_stuck(colored_mdp, node.parameter_space, mdp, policy_fixed):
        logger.warning("policy should be SAT but some member does not offer one of its actions")
    if node.parameter_space.size == 1:
        paynt.mdp_family._utils.assert_mdp_is_deterministic(colored_mdp, mdp, node.parameter_space)
    DOUBLE_CHECK_PRECISION = 1e-6
    default_precision = Property.model_checking_precision
    Property.set_model_checking_precision(DOUBLE_CHECK_PRECISION)
    policy_result = mdp.model_check_property(prop, alt=True)
    Property.set_model_checking_precision(default_precision)
    if not policy_result.sat:
        logger.warning(f"policy should be SAT but (most likely due to model checking precision) has value {policy_result.value}")
    return


class PolicyTreeNode(paynt.synthesizer.search_node.SearchNode):
    # bumped by make_policies_compatible; reset (and read) only by PolicyTree.postprocess
    mdps_model_checked: int = 0

    def __init__(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, parent_info: paynt.synthesizer.search_node.ParentInfo | None = None
    ):
        super().__init__(parameter_space, parent_info)

        # a previously-computed game-abstraction policy supplied by the parent split, tried before solving a
        # fresh game abstraction for this node (see PolicyTreeSynthesizer.verify_parameter_space)
        self.candidate_policy: list[int | None] | None = None

        self.splitter: int | None = None
        self.suboptions: list[list[int]] = []
        self.child_nodes: list[PolicyTreeNode] = []

        self.sat: bool | None = None
        self.policy_index: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.sat is not None

    def num_nodes(self) -> int:
        num = 1
        for child in self.child_nodes:
            num += child.num_nodes()
        return num

    def num_leaves(self) -> int:
        num = 1 if self.is_leaf else 0
        for child in self.child_nodes:
            num += child.num_leaves()
        return num

    def attach_children(
        self,
        splitter: int,
        suboptions: list[list[int]],
        parameter_subspaces: list[paynt.parameter_space.parameter_space.ParameterSpace],
        candidate_policies: list[list[int | None] | None] | None = None,
    ) -> None:
        """Wrap each of parameter_subspaces (already-split ParameterSpace values) as a child PolicyTreeNode."""
        self.splitter = splitter
        self.suboptions = suboptions
        self.child_nodes = []
        if candidate_policies is None:
            candidate_policies = [None for _ in parameter_subspaces]
        for parameter_subspace, candidate_policy in zip(parameter_subspaces, candidate_policies, strict=False):
            child_node = PolicyTreeNode(parameter_subspace)
            child_node.candidate_policy = candidate_policy
            self.child_nodes.append(child_node)

    def double_check(self, colored_mdp: paynt.colored_mdp.ColoredMdp, prop: Property, policies: list[Policy | None]) -> None:
        assert self.sat is not None
        self.mdp, self.selected_choices = colored_mdp.build(self.parameter_space)
        if self.sat is False:
            result = self.mdp.model_check_property(prop)
            assert not result.sat
        else:
            assert self.policy_index is not None
            policy = policies[self.policy_index]
            assert policy is not None
            double_check_policy(colored_mdp, self, prop, policy[0])

    def merge_children_indices(self, indices: list[int]) -> None:
        if len(indices) <= 1:
            return
        target = indices[0]
        for j in reversed(indices[1:]):
            self.suboptions[target] += self.suboptions[j]
            assert self.splitter is not None
            self.child_nodes[target].parameter_space.parameter_set_options(self.splitter, self.suboptions[target])
            # the choices of the members of the merged node: those of either
            if self.child_nodes[target].selected_choices is not None and self.child_nodes[j].selected_choices is not None:
                self.child_nodes[target].selected_choices = self.child_nodes[target].selected_choices | self.child_nodes[j].selected_choices
            self.suboptions.pop(j)
            self.child_nodes.pop(j)

        if len(self.child_nodes) > 1:
            return
        # a single child remains, can be merged into parent
        child_node = self.child_nodes[0]
        self.sat = child_node.sat
        self.policy_index = child_node.policy_index
        self.splitter = None
        self.suboptions = []
        self.child_nodes = []

    def merge_children_having_same_solution(self) -> None:
        if self.is_leaf:
            return

        # merge UNSAT children
        indices = [i for i, child in enumerate(self.child_nodes) if child.sat is False]
        self.merge_children_indices(indices)

        # merge children having the same policy
        i = 0
        while i < len(self.child_nodes):
            child1 = self.child_nodes[i]
            if child1.sat is not True:
                i += 1
                continue

            join_to_i = [i]
            # collect other children to merge to i
            for j in range(i + 1, len(self.child_nodes)):
                child2 = self.child_nodes[j]
                if child2.policy_index == child1.policy_index:
                    join_to_i.append(j)
            self.merge_children_indices(join_to_i)
            i += 1

    @staticmethod
    def make_policies_compatible(
        colored_mdp: paynt.colored_mdp.ColoredMdp, prop: Property, node1: PolicyTreeNode, node2: PolicyTreeNode, policies: list[Policy | None]
    ) -> Policy | None:
        assert node1.policy_index is not None and node2.policy_index is not None
        policy1 = policies[node1.policy_index]
        policy2 = policies[node2.policy_index]
        assert policy1 is not None and policy2 is not None
        policy = merge_policies(policy1, policy2)
        if policy is not None:
            return policy

        policy12, policy21 = merge_policies_exclusively(policy1, policy2)
        # try policy1 for node2's parameter space, then policy2 for node1's
        for policy_ext, policy_own, node in ((policy12, policy1, node2), (policy21, policy2, node1)):
            PolicyTreeNode.mdps_model_checked += 1
            policy = PolicyTreeNode.extend_to(colored_mdp, prop, node, policy_ext, policy_own)
            if policy is not None:
                return policy
        # neither fits
        return None

    @staticmethod
    def extend_to(
        colored_mdp: paynt.colored_mdp.ColoredMdp, prop: Property, node: PolicyTreeNode, policy: list[int | None], policy_own: Policy
    ) -> Policy | None:
        """The policy, extending policy_own (the policy of the other node, whose actions it takes where that one has any), for the members of node too, if it
        satisfies prop in all of them: it offers them its action wherever it goes. None if it does not."""
        try:
            (policy_fixed, _), mdp = paynt.mdp_family._utils.fix_and_apply_policy_to_parameter_space(colored_mdp, node.selected_choices, policy)
        except paynt.utils.error_handling.StateWithoutChoice:
            # an action of the policy that no member of the node offers
            return None
        if paynt.mdp_family._utils.policy_gets_stuck(colored_mdp, node.parameter_space, mdp, policy_fixed):
            return None
        if not mdp.model_check_property(prop, alt=True).sat:
            return None
        # where the node's members do not go, the actions for the other node's
        actions = [action if action is not None else policy_own[0][state] for state, action in enumerate(policy_fixed)]
        return (actions, [state for state, action in enumerate(actions) if action is not None])

    def merge_children_having_compatible_policies(self, colored_mdp: paynt.colored_mdp.ColoredMdp, prop: Property, policies: list[Policy | None]) -> None:
        if self.is_leaf:
            return
        i = 0
        while i < len(self.child_nodes):
            child1 = self.child_nodes[i]
            if child1.sat is not True:
                i += 1
                continue

            j = i + 1
            while j < len(self.child_nodes):
                child2 = self.child_nodes[j]
                if child2.sat is not True:
                    j += 1
                    continue
                policy = PolicyTreeNode.make_policies_compatible(colored_mdp, prop, child1, child2, policies)
                if policy is None:
                    j += 1
                    continue
                # nodes can be merged
                assert child1.policy_index is not None and child2.policy_index is not None
                policies[child1.policy_index] = policy
                policies[child2.policy_index] = None
                # merged right away: the next sibling is checked against every member of child1 so far
                self.merge_children_indices([i, j])
            i += 1

    def skip_redundant_children(self) -> None:
        """Adopt grandchildren of each child that uses the same splitter as self."""
        if self.splitter is None:
            return
        suboptions = []
        child_nodes = []
        for child_index, child in enumerate(self.child_nodes):
            if child.splitter != self.splitter:
                suboptions.append(self.suboptions[child_index])
                child_nodes.append(self.child_nodes[child_index])
            else:
                for grandchild_index, grandchild in enumerate(child.child_nodes):
                    suboptions.append(child.suboptions[grandchild_index])
                    child_nodes.append(grandchild)
        self.suboptions = suboptions
        self.child_nodes = child_nodes

    @property
    def node_id(self) -> str:
        return str(self.parameter_space).replace(" ", "").replace(":", "=")

    def add_nodes_to_graphviz_tree(self, graphviz_tree: graphviz.Digraph) -> None:
        node_label = ""
        if self.sat is False:
            node_label = "∅"
            # node_label = "X"
        elif self.sat is True:
            # node_label = "✓"
            node_label = f"p{self.policy_index}"
        graphviz_tree.node(self.node_id, label=node_label, shape="ellipse", width="0.15", height="0.15")
        # enumerating in reverse to print policies in ascending order, from left to right
        for child in reversed(self.child_nodes):
            child.add_nodes_to_graphviz_tree(graphviz_tree)

    def add_edges_to_graphviz_tree(self, graphviz_tree: graphviz.Digraph) -> None:
        if self.splitter is None:
            return
        self.parameter_space.parameter_name(self.splitter)
        for index, child in enumerate(self.child_nodes):
            edge_label = self.parameter_space.parameter_options_to_string(self.splitter, self.suboptions[index])
            graphviz_tree.edge(self.node_id, child.node_id, label=edge_label)
            child.add_edges_to_graphviz_tree(graphviz_tree)


class PolicyTree:
    def __init__(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace):
        self.root = PolicyTreeNode(parameter_space)
        self.policies: list[Policy | None] = []

    def new_policy(self, policy: list[int | None]) -> int:
        policy_index = len(self.policies)
        mask = [state for state, action in enumerate(policy) if action is not None]
        self.policies.append((policy, mask))
        return policy_index

    def collect_all(self) -> list[PolicyTreeNode]:
        node_queue = [self.root]
        all_nodes = []
        while node_queue:
            node = node_queue.pop(0)
            all_nodes.append(node)
            node_queue += node.child_nodes
        return all_nodes

    def collect_leaves(self) -> list[PolicyTreeNode]:
        node_queue = [self.root]
        leaves = []
        while node_queue:
            node = node_queue.pop(0)
            if node.is_leaf:
                leaves.append(node)
            else:
                node_queue = node_queue + node.child_nodes
        return leaves

    def collect_nonleaves(self) -> list[PolicyTreeNode]:
        node_queue = [self.root]
        nonleaves = []
        while node_queue:
            node = node_queue.pop(0)
            if not node.is_leaf:
                nonleaves.append(node)
                node_queue = node_queue + node.child_nodes
        return nonleaves

    def collect_sat(self) -> list[PolicyTreeNode]:
        node_queue = [self.root]
        sat = []
        while node_queue:
            node = node_queue.pop(0)
            if node.sat:
                sat.append(node)
            else:
                node_queue += node.child_nodes
        return sat

    def double_check(self, colored_mdp: paynt.colored_mdp.ColoredMdp, prop: Property) -> None:
        leaves = self.collect_leaves()
        logger.info(f"double-checking {len(leaves)} parameter spaces...")
        for leaf in leaves:
            leaf.double_check(colored_mdp, prop, self.policies)
        logger.info("all solutions are OK")

    def print_stats(self) -> None:
        members_total = self.root.parameter_space.size
        num_policies = len(self.policies)

        members_satisfied = 0
        num_leaves_singleton = 0
        leaves = self.collect_leaves()
        for node in leaves:
            if node.parameter_space.size == 1:
                num_leaves_singleton += 1
            if node.sat:
                members_satisfied += node.parameter_space.size
        satisfied_percentage = round(members_satisfied / members_total * 100, 0)
        members_unsatisfied = members_total - members_satisfied

        num_nodes = len(self.collect_all())
        num_leaves = len(leaves)
        num_leaves_solvable = len(self.collect_sat())
        num_leaves_unsolvable = num_leaves - num_leaves_solvable
        leaf_solvable_avg: float | str
        leaf_unsolvable_avg: float | str
        if num_leaves_solvable > 0:
            leaf_solvable_avg = round(members_satisfied / num_leaves_solvable, 1)
        else:
            leaf_solvable_avg = "NA"

        if num_leaves_unsolvable > 0:
            leaf_unsolvable_avg = round(members_unsatisfied / num_leaves_unsolvable, 1)
        else:
            leaf_unsolvable_avg = "NA"

        logger.info("--------------------")
        logger.info("Policy tree summary:")
        logger.info(
            "found {} satisfying {} for {}/{} family members ({}%)".format(
                num_policies, "policy" if num_policies == 1 else "policies", members_satisfied, members_total, satisfied_percentage
            )
        )
        logger.info(f"policy tree has {num_nodes} nodes, {num_leaves} of them are leaves:")
        logger.info(f"\t  solvable leaves: {num_leaves_solvable} (avg.size: {leaf_solvable_avg})")
        logger.info(f"\tunsolvable leaves: {num_leaves_unsolvable} (avg.size: {leaf_unsolvable_avg})")
        logger.info(f"\t singleton leaves: {num_leaves_singleton}")

        logger.info("--------------------")

    def discard_unused_policies(self) -> None:
        policy_old_to_new: list[int | None] = [None for _ in self.policies]
        num_policies = 0
        for policy_index, policy in enumerate(self.policies):
            if policy is not None:
                policy_old_to_new[policy_index] = num_policies
                num_policies += 1
        self.policies = [policy for policy in self.policies if policy is not None]
        assert num_policies == len(self.policies)
        for leaf in self.collect_sat():
            assert leaf.policy_index is not None
            leaf.policy_index = policy_old_to_new[leaf.policy_index]
            assert leaf.policy_index is not None

    def merge_compatible_policies(self, policy_indices: list[int]) -> list[int]:
        policy_old_to_new_map = [policy_index for policy_index, _ in enumerate(self.policies)]

        for policy1_index_index, policy1_index in enumerate(policy_indices):
            policy1 = self.policies[policy1_index]
            if policy1 is None:
                continue
            for policy2_index in policy_indices[policy1_index_index + 1 :]:
                policy2 = self.policies[policy2_index]
                if policy2 is None:
                    continue
                policy = merge_policies(policy1, policy2)
                if policy is None:
                    continue
                # store updated policy
                self.policies[policy1_index] = policy
                policy1 = policy
                # discard irrelevant policy
                policy_old_to_new_map[policy2_index] = policy1_index
                self.policies[policy2_index] = None

        return policy_old_to_new_map

    def postprocess(self, colored_mdp: paynt.colored_mdp.ColoredMdp, prop: Property) -> int:

        postprocessing_timer = paynt.utils.timer.Timer()
        postprocessing_timer.start()
        logger.info("post-processing the policy tree...")

        logger.info("merging SAT siblings solved by non-exclusively compatible policies...")
        PolicyTreeNode.mdps_model_checked = 0
        nodes_before = self.root.num_nodes()
        for node in reversed(self.collect_all()):
            node.merge_children_having_compatible_policies(colored_mdp, prop, self.policies)
        self.discard_unused_policies()
        nodes_removed = nodes_before - self.root.num_nodes()
        logger.info(f"additional {PolicyTreeNode.mdps_model_checked} MDPs were model checked")
        logger.info(f"removed {nodes_removed} nodes")

        logger.info("merging all exclusively compatible policies...")
        policies_before = len(self.policies)
        policy_indices = [index for index, _ in enumerate(self.policies)]
        policy_old_to_new_map = self.merge_compatible_policies(policy_indices)
        for leaf in self.collect_sat():
            assert leaf.policy_index is not None
            leaf.policy_index = policy_old_to_new_map[leaf.policy_index]
        self.discard_unused_policies()
        policies_removed = policies_before - len(self.policies)
        logger.info(f"removed {policies_removed} policies")

        logger.info("reducing tree height...")
        nodes_before = self.root.num_nodes()
        for node in reversed(self.collect_nonleaves()):
            node.skip_redundant_children()
        nodes_removed = nodes_before - self.root.num_nodes()
        logger.info(f"removed {nodes_removed} nodes")

        logger.info("merging siblings that have the same solution...")
        nodes_before = self.root.num_nodes()
        for node in reversed(self.collect_nonleaves()):
            node.merge_children_having_same_solution()
        nodes_removed = nodes_before - self.root.num_nodes()
        logger.info(f"removed {nodes_removed} nodes")

        postprocessing_timer.stop()
        time = int(postprocessing_timer.read())
        logger.debug(f"postprocessing took {time} s")
        return time

    def extract_policies(self, colored_mdp: paynt.colored_mdp.ColoredMdp) -> dict[str, list[tuple[dict[str, Any], str]]]:
        policies = {}
        for policy_index, policy in enumerate(self.policies):
            assert policy is not None
            policies[f"p{policy_index}"] = paynt.mdp_family._utils.policy_to_state_valuation_actions(colored_mdp, policy)
        return policies

    def extract_policy_tree(self, colored_mdp: paynt.colored_mdp.ColoredMdp) -> graphviz.Digraph:
        logging.getLogger("graphviz").setLevel(logging.WARNING)
        logging.getLogger("graphviz.sources").setLevel(logging.ERROR)
        graphviz_tree = graphviz.Digraph(comment="policy_tree")
        self.root.add_nodes_to_graphviz_tree(graphviz_tree)
        self.root.add_edges_to_graphviz_tree(graphviz_tree)
        return graphviz_tree
