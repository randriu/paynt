from __future__ import annotations

from typing import Any

import paynt.colored_mdp
from paynt.dt._utils import DtInfo, DONT_CARE_ACTION_LABEL
import paynt.dt.task
import paynt.parameter_space.parameter_space
import paynt.model.model

from paynt.parser._utils import make_rewards_action_based

import stormpy
import payntbind

from .decision_tree import DecisionTree, DtVariable
from .coloring_general import decision_tree_coloring
from ._utils import get_state_valuations

import logging

logger = logging.getLogger(__name__)


class DtColoredMdpFactory:
    """Constructs a ColoredMdp (feature_kind "dt") for a given decision-tree depth."""

    feature_kind = "dt"

    # label for action executing a random action selection
    DONT_CARE_ACTION_LABEL = DONT_CARE_ACTION_LABEL
    # if true, irrelevant states will not be considered for tree mapping
    filter_deterministic_states = True

    def __init__(self, mdp: Any, build_task: paynt.dt.task.DtTask | None = None, use_exact: bool = False):
        self.build_task = build_task
        self.use_exact = use_exact
        # build_task is optional here (see class docstring) -- getattr falls back to DtTask's own
        # default when it is None
        add_dont_care_action = getattr(build_task, "add_dont_care_action", True)

        make_rewards_action_based(mdp)  # needed for initialization

        # identify relevant states: non-absorbing states with more than one action
        state_is_relevant = [True for state in range(mdp.nr_states)]
        state_is_absorbing = paynt.model.model.ModelIndex.identify_absorbing_states(mdp)
        state_is_relevant = [relevant and not state_is_absorbing[state] for state, relevant in enumerate(state_is_relevant)]

        if DtColoredMdpFactory.filter_deterministic_states:
            state_has_actions = paynt.model.model.ModelIndex.identify_states_with_actions(mdp)
            state_is_relevant = [relevant and state_has_actions[state] for state, relevant in enumerate(state_is_relevant)]
        state_is_relevant_bv = stormpy.BitVector(mdp.nr_states)
        [state_is_relevant_bv.set(state, value) for state, value in enumerate(state_is_relevant)]
        logger.debug(f"MDP has {state_is_relevant_bv.number_of_set_bits()}/{state_is_relevant_bv.size()} relevant states")
        self.state_is_relevant = state_is_relevant
        self.state_is_relevant_bv = state_is_relevant_bv

        action_labels, _ = payntbind.synthesis.extractActionLabels(mdp)
        if DtColoredMdpFactory.DONT_CARE_ACTION_LABEL not in action_labels and add_dont_care_action:
            logger.debug("adding explicit don't-care action to relevant states...")
            mdp = payntbind.synthesis.addDontCareAction(mdp, self.state_is_relevant_bv)

        self.underlying_mdp = mdp
        self.choice_destinations = payntbind.synthesis.computeChoiceDestinations(mdp)
        self.action_labels, self.choice_to_action = payntbind.synthesis.extractActionLabels(mdp)
        logger.info(f"MDP has {len(self.action_labels)} actions")
        # TODO filter irrelevant actions?

        # get variable domains on relevant states
        variable_name, state_valuations = get_state_valuations(mdp)
        num_variables = len(variable_name)
        variable_domain_sets: list[set[Any]] = [set() for variable in range(num_variables)]
        for state in self.state_is_relevant_bv:
            valuation = state_valuations[state]
            for variable in range(num_variables):
                variable_domain_sets[variable].add(valuation[variable])
        variable_domain = [sorted(domain) for domain in variable_domain_sets]

        # filter variables having only one option
        variable_mask = [len(domain) > 1 for domain in variable_domain]
        variable_name = [value for variable, value in enumerate(variable_name) if variable_mask[variable]]
        variable_domain = [value for variable, value in enumerate(variable_domain) if variable_mask[variable]]
        # we filter unused variables from state valuations: this means that multiple states can now have the same "valuation"
        state_valuations = [[value for variable, value in enumerate(valuations) if variable_mask[variable]] for valuations in state_valuations]

        # DtVariable's own domain parameter is typed as set[int], but its __init__ actually iterates and
        # sorts whatever iterable it's given -- a plain sorted list (built above) works fine at runtime
        self.variables = [DtVariable(name, variable_domain[variable]) for variable, name in enumerate(variable_name)]  # type: ignore[arg-type]
        self.relevant_state_valuations = state_valuations
        logger.debug(f"found the following {len(self.variables)} variables: {[str(v) for v in self.variables]}")

    def build(self) -> paynt.colored_mdp.ColoredMdp:
        """Produce a ColoredMdp at build_task's own default tree depth."""
        assert self.build_task is not None
        return self.reset_tree(self.build_task.tree_depth)

    def reset_tree(self, depth: int, enable_harmonization: bool = True, general: bool = False) -> paynt.colored_mdp.ColoredMdp:
        """Produce a ColoredMdp at the given tree depth, discarding any previous tree and coloring.

        :param enable_harmonization: whether ColoringSmt's areChoicesConsistent also proposes a split (harmonization) for an inconsistent scheduler. Only AR
            reads that split; a caller that merely wants the verdict (mapping a scheduler) turns it off, which spares the coloring the work of computing it. The
            general coloring never proposes one, so it ignores this.
        :param general: if True, build the tree's coloring as a payntbind.synthesis.ColoringGeneral (see paynt.dt.coloring_general.decision_tree_coloring)
            instead of the default ColoringSmt. Everything else about the produced ColoredMdp -- its ParameterSpace, DtInfo, feature_kind -- is unaffected: the
            two colorings are built from identical inputs and expose an identical parameter layout (see decision_tree_coloring's own docstring), so dtpaynt's
            default (general=False) path and behavior are completely untouched by this option.
        """
        num_actions = len(self.action_labels)
        dont_care_action = num_actions
        if DtColoredMdpFactory.DONT_CARE_ACTION_LABEL in self.action_labels:
            dont_care_action = self.action_labels.index(DtColoredMdpFactory.DONT_CARE_ACTION_LABEL)

        decision_tree = DecisionTree(self.action_labels, self.variables)
        decision_tree.set_depth(depth)

        variables = decision_tree.variables
        variable_name = [v.name for v in variables]
        variable_domain = [v.domain for v in variables]
        if general:
            coloring, parameter_info = decision_tree_coloring(
                self.underlying_mdp.nondeterministic_choice_indices,
                self.choice_to_action,
                dont_care_action,
                self.action_labels,
                variables,
                self.relevant_state_valuations,
                self.state_is_relevant_bv,
                decision_tree,
            )
            initial_state = self.underlying_mdp.initial_states[0]
            coloring.enableStateExploration(initial_state, self.choice_destinations)
        else:
            tree_list = decision_tree.to_list()
            coloring = payntbind.synthesis.ColoringSmt(
                self.underlying_mdp.nondeterministic_choice_indices,
                self.choice_to_action,
                num_actions,
                dont_care_action,
                self.underlying_mdp.state_valuations,
                self.state_is_relevant_bv,
                variable_name,
                variable_domain,
                tree_list,
                enable_harmonization,
            )
            coloring.enableStateExploration(self.underlying_mdp)
            parameter_info = coloring.getFamilyInfo()

        # reconstruct the parameter space
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        is_action_parameter = [False for _ in parameter_info]
        is_decision_parameter = [False for _ in parameter_info]
        is_variable_parameter = [False for _ in parameter_info]
        node_parameter_info: list[list[tuple[int, str, str]]] = [[] for _ in decision_tree.collect_nodes()]
        for parameter_id, info in enumerate(parameter_info):
            node, parameter_name, parameter_type = info
            node_parameter_info[node].append((parameter_id, parameter_name, parameter_type))
            if parameter_type == "__action__":
                is_action_parameter[parameter_id] = True
                option_labels = self.action_labels
            elif parameter_type == "__decision__":
                is_decision_parameter[parameter_id] = True
                option_labels = variable_name
            else:
                is_variable_parameter[parameter_id] = True
                variable = variable_name.index(parameter_type)
                option_labels = variables[variable].parameter_domain
            parameter_space.add_parameter(parameter_name, option_labels)
        decision_tree.root.associate_parameters(node_parameter_info)

        colored_mdp = paynt.colored_mdp.ColoredMdp(self.underlying_mdp, parameter_space, coloring, self.use_exact, feature_kind="dt")
        colored_mdp.feature_info = DtInfo(
            action_labels=self.action_labels,
            choice_to_action=self.choice_to_action,
            state_is_relevant=self.state_is_relevant,
            state_is_relevant_bv=self.state_is_relevant_bv,
            variables=self.variables,
            relevant_state_valuations=self.relevant_state_valuations,
            decision_tree=decision_tree,
            is_action_parameter=is_action_parameter,
            is_decision_parameter=is_decision_parameter,
            is_variable_parameter=is_variable_parameter,
        )
        return colored_mdp
