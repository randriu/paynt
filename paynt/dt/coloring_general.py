"""Decision trees as a general coloring (see paynt/utils/coloring_builder.py): the same formula ColoringSmt encodes as a
dedicated, DT-specialized Z3 tree walk, reproduced here as an ordinary z3py template over ColoringBuilder --
so SMPMC (and CEGIS/OneByOne) can run directly on decision-tree synthesis. ColoringSmt itself is untouched: this is
an alternative coloring for the same DtColoredMdpFactory/DtInfo/ParameterSpace shape (opted into per depth via
reset_tree(general=True)), never a replacement for it.

Parameter layout matches ColoringSmt's own (see ColoringSmt.getFamilyInfo/TreeNode::createHoles) exactly, so
DtColoredMdpFactory.reset_tree can reuse its existing parameter-space/DtInfo reconstruction unchanged: per inner
node, V_<id> (the decision variable: one option per tree variable) then <var>_<id> per variable (the threshold:
options = that variable's domain except its own maximum, see DtVariable.parameter_domain); per leaf, A_<id> (the
action, one option per action label). Node identifiers -- and so parameter order -- follow DecisionTree's own
pre-order, true-child-first traversal (DecisionTreeNode.assign_identifiers).

Color, for a relevant state s and a choice c with action a = choice_to_action[c] (positive form, which ColoringGeneral's
three-valued evaluation decides exactly, since the steps along one path use disjoint parameters):

    Or_{leaf p} ( taken_p(x_s) AND ( A_p == a OR (a == dont_care AND in_bits(A_p, unavailable_s)) ) )

where unavailable_s is the bitmask of actions not available at s (so the don't-care choice is also enabled when a
leaf's action doesn't exist at s, as in ColoringSmt).

taken_p is the AND, over each inner node n on p's root-to-leaf path, of that node's *step*:

    true step:  Or_v ( V_n == v AND x_{s,v} <= T_{n,v} )
    false step: Or_v ( V_n == v AND x_{s,v} >  T_{n,v} )

(x_s and T are both plain option indices into that variable's own domain, exactly as ColoringSmt stores them. No
boundary special-casing is needed, unlike ColoringSmt.cpp's own encoding: comparing a state's single data value against
a threshold parameter's domain is already exact for the contiguous threshold domains tree synthesis produces, and
equalities against a bare parameter (V_n == v, A_p == a) are exact for any domain.)

Irrelevant states are handled by the engine itself (ColoringGeneral's state_is_relevant, "enable the first choice
in base_choices"), exactly like ColoringSmt -- this template says nothing about them.
"""

from __future__ import annotations

from typing import Any

import z3

import paynt.dt.decision_tree
import paynt.utils.coloring_builder

import logging

logger = logging.getLogger(__name__)

#: The highest action index that can be unavailable at a state: the actions unavailable at a state are kept in one signed 64-bit data word (see
#: ColoringBuilder.in_bits), whose sign bit is left out.
_MAX_UNAVAILABLE_ACTION_INDEX = 62


def _state_option_indices(
    relevant_state_valuations: list[list[int]], state_is_relevant_bv: Any, variables: list[paynt.dt.decision_tree.DtVariable]
) -> list[list[int]]:
    """Convert each state's raw variable valuation into option indices (its position in that variable's own domain), matching ColoringSmt's own internal
    conversion (ColoringSmt.cpp's state_valuation array).

    Irrelevant states get an all-zero placeholder: never read (state_is_relevant gates evaluation), and their raw
    valuation may not even appear in a domain built only from relevant states (see DtColoredMdpFactory.__init__).
    """
    value_to_index = [{value: index for index, value in enumerate(variable.domain)} for variable in variables]
    num_variables = len(variables)
    result = []
    for state, valuation in enumerate(relevant_state_valuations):
        if state_is_relevant_bv[state]:
            result.append([value_to_index[variable][valuation[variable]] for variable in range(num_variables)])
        else:
            result.append([0] * num_variables)
    return result


def decision_tree_coloring(
    row_groups: list[int],
    choice_to_action: list[int],
    dont_care_action: int,
    action_labels: list[str],
    variables: list[paynt.dt.decision_tree.DtVariable],
    relevant_state_valuations: list[list[int]],
    state_is_relevant_bv: Any,
    decision_tree: paynt.dt.decision_tree.DecisionTree,
) -> tuple[Any, list[tuple[int, str, str]]]:
    """Build a ColoringGeneral for decision_tree, over the same underlying MDP DtColoredMdpFactory.reset_tree builds ColoringSmt from -- same inputs (see
    reset_tree), so both colorings see identical data.

    :returns: (coloring, parameter_info) -- parameter_info in ColoringSmt.getFamilyInfo()'s own format (one (node_identifier, parameter_name, parameter_type)
        tuple per parameter, parameter_type one of "__decision__"/"__action__"/a variable's own name), so reset_tree's existing parameter-space reconstruction
        applies completely unchanged.
    """
    num_states = len(row_groups) - 1
    num_actions = len(action_labels)
    num_variables = len(variables)
    dont_care_defined = dont_care_action < num_actions

    state_option_index = _state_option_indices(relevant_state_valuations, state_is_relevant_bv, variables)

    # per-state data: the option-index valuation columns, plus (only if a don't-care action exists) the bitmask of
    # actions NOT available at that state -- mirrors ColoringSmt's state_available_actions
    state_available_mask = [0] * num_states
    for state in range(num_states):
        for choice in range(row_groups[state], row_groups[state + 1]):
            state_available_mask[state] |= 1 << choice_to_action[choice]
    all_actions_mask = (1 << num_actions) - 1
    # an irrelevant state is never evaluated, and the don't-care action only exists at relevant states: its word stays empty, or that action would count
    # as unavailable there
    state_unavailable_mask = [((~mask) & all_actions_mask) if state_is_relevant_bv[state] else 0 for state, mask in enumerate(state_available_mask)]
    if dont_care_defined:
        highest_unavailable = max((mask.bit_length() for mask in state_unavailable_mask), default=0) - 1
        if highest_unavailable > _MAX_UNAVAILABLE_ACTION_INDEX:
            raise ValueError(
                f"the general coloring of decision trees keeps the actions unavailable at a state in one 64-bit word, which holds action indices up to "
                f"{_MAX_UNAVAILABLE_ACTION_INDEX}, but action {action_labels[highest_unavailable]!r} (index {highest_unavailable} of {num_actions}) is "
                "unavailable at some state: use --method ar for this model"
            )

    # Parameters are allocated in the exact order ColoringSmt's createHoles assigns hole ids: a pre-order,
    # true-child-first walk of the tree, allocating (decision, then one threshold per variable) at an inner node, or
    # (one action parameter) at a leaf. This is a pure tree walk -- independent of ColoringBuilder -- so the
    # resulting parameter count is known before the builder (which needs it up front) is even constructed.
    parameter_info: list[tuple[int, str, str]] = []

    def allocate(count: int, identifier: int, name: str, kind: str) -> list[int]:
        start = len(parameter_info)
        for _ in range(count):
            parameter_info.append((identifier, name, kind))
        return list(range(start, start + count))

    node_decision_parameter: dict[int, int] = {}
    node_threshold_parameters: dict[int, list[int]] = {}
    node_action_parameter: dict[int, int] = {}

    def allocate_node(node: paynt.dt.decision_tree.DecisionTreeNode) -> None:
        assert node.identifier is not None
        if node.is_terminal:
            [action_parameter] = allocate(1, node.identifier, f"A_{node.identifier}", "__action__")
            node_action_parameter[node.identifier] = action_parameter
            return
        [decision_parameter] = allocate(1, node.identifier, f"V_{node.identifier}", "__decision__")
        node_decision_parameter[node.identifier] = decision_parameter
        threshold_parameters = []
        for variable in range(num_variables):
            name = variables[variable].name
            [threshold_parameter] = allocate(1, node.identifier, f"{name}_{node.identifier}", name)
            threshold_parameters.append(threshold_parameter)
        node_threshold_parameters[node.identifier] = threshold_parameters
        assert node.child_true is not None and node.child_false is not None
        allocate_node(node.child_true)
        allocate_node(node.child_false)

    allocate_node(decision_tree.root)

    builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, len(parameter_info), state_is_relevant=state_is_relevant_bv)
    parameters = builder.parameters
    state_valuation_cols = [builder.state_column([state_option_index[state][v] for state in range(num_states)]) for v in range(num_variables)]
    unavailable_col = builder.state_column(state_unavailable_mask) if dont_care_defined else None
    action_col = builder.choice_column(choice_to_action)

    def leaf_paths(node: paynt.dt.decision_tree.DecisionTreeNode, prefix: list[tuple[int, bool]]) -> list[tuple[int, list[tuple[int, bool]]]]:
        """All (leaf identifier, path) pairs below node, path = [(inner node identifier, went_true), ...]."""
        if node.is_terminal:
            assert node.identifier is not None
            return [(node.identifier, prefix)]
        assert node.child_true is not None and node.child_false is not None and node.identifier is not None
        return leaf_paths(node.child_true, prefix + [(node.identifier, True)]) + leaf_paths(node.child_false, prefix + [(node.identifier, False)])

    def step(inner_identifier: int, went_true: bool, state_valuation: list[Any]) -> Any:
        decision_var = parameters[node_decision_parameter[inner_identifier]]
        thresholds = [parameters[p] for p in node_threshold_parameters[inner_identifier]]
        disjuncts = []
        for variable in range(num_variables):
            comparison = state_valuation[variable] <= thresholds[variable] if went_true else state_valuation[variable] > thresholds[variable]
            disjuncts.append(z3.And(decision_var == variable, comparison))
        return z3.Or(*disjuncts) if disjuncts else z3.BoolVal(False)

    def taken(path: list[tuple[int, bool]], state_valuation: list[Any]) -> Any:
        conjuncts = [step(identifier, went_true, state_valuation) for identifier, went_true in path]
        return z3.And(*conjuncts) if conjuncts else z3.BoolVal(True)

    root_leaf_paths = leaf_paths(decision_tree.root, [])

    # A SINGLE template for every relevant state's every choice: state_valuation/action_col/is_dont_care_col/
    # unavailable_col are column *references* (StateColTerm/ChoiceColTerm node ids), not per-state/per-choice
    # constants -- ColoringGeneral resolves them against the current state/choice row at evaluation time, so the
    # same compiled formula is correct for every (state, choice) it's attached to; building it once (rather than
    # once per state) keeps compilation O(tree size), not O(states * tree size).
    state_valuation = state_valuation_cols
    disjuncts = []
    for leaf_identifier, path in root_leaf_paths:
        action_parameter = parameters[node_action_parameter[leaf_identifier]]
        leaf_taken = taken(path, state_valuation)
        direct = action_parameter == action_col
        if dont_care_defined:
            assert unavailable_col is not None
            fallback = z3.And(action_col == dont_care_action, builder.in_bits(action_parameter, unavailable_col))
            disjuncts.append(z3.And(leaf_taken, z3.Or(direct, fallback)))
        else:
            disjuncts.append(z3.And(leaf_taken, direct))
    template = builder.template(z3.Or(*disjuncts) if disjuncts else z3.BoolVal(False))

    for state in range(num_states):
        if state_is_relevant_bv[state]:
            builder.color(range(row_groups[state], row_groups[state + 1]), template)

    coloring = builder.build()
    return coloring, parameter_info
