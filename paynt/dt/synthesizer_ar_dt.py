import paynt.synthesizer.synthesizer_ar
import paynt.synthesizer.search_node
import paynt.utils.scoring
import paynt.specification.property_result

import logging

logger = logging.getLogger(__name__)


class DtSearchNode(paynt.synthesizer.search_node.SearchNode):
    def __init__(self, parameter_space, parent_info=None):
        super().__init__(parameter_space, parent_info)
        # set only by ColoredMdp.scheduler_is_consistent's "dt" branch, and only for a single-property specification --
        # None otherwise, giving the "scheduler preserved across split" shortcut in verify_parameter_space a
        # real declared default rather than the getattr(..., None) it used to need
        self.scheduler_choices = None


class SynthesizerARDt(paynt.synthesizer.synthesizer_ar.SynthesizerAR):
    """AR specialized for decision-tree synthesis: splits by parameter kind (action/decision/variable) rather than by scored inconsistency variance, and adds
    harmonization.

    This is the inner search engine; the outer DtSynthesizer (paynt.dt.synthesizer) constructs a fresh instance of this class for every tree depth it tries.
    """

    search_node_type = DtSearchNode

    def __init__(self, colored_mdp, task):
        super().__init__(colored_mdp, task)
        self.counters_reset()

    @property
    def method_name(self):
        return "AR (decision tree)"

    def verify_parameter_selection(self, node, parameter_selection):
        spec = self.task.specification
        assignment = node.parameter_space.assume_options_copy(parameter_selection)
        dtmc = self.colored_mdp.build_assignment(assignment)
        res = dtmc.check_specification(spec)
        if not res.constraints_result.sat:
            return
        if not spec.has_optimality:
            node.analysis_result.improving_assignment = assignment
            node.analysis_result.can_improve = False
            return
        assignment_value = res.optimality_result.value
        if spec.optimality.improves_optimum(assignment_value):
            # logger.info(f"harmonization achieved value {res.optimality_result.value}")
            self.num_harmonization_succeeded += 1
            node.analysis_result.improving_assignment = assignment
            node.analysis_result.improving_value = assignment_value
            node.analysis_result.can_improve = True
            self.update_optimum(node)

    def harmonize_inconsistent_scheduler(self, node):
        """Try forcing the harmonizing parameter (the first one still locally inconsistent in some undecided property's own scheduler) to each of its two
        inconsistent options in turn, hoping one of the two concrete assignments happens to satisfy the whole spec outright.

        A pure-L(h) node -- every remaining property's own scheduler is already locally consistent, but they disagree with each other -- has nothing here to
        harmonize against, so this is a no-op in that case: harmonization is a speed-up attempt, not something split_undecided_space depends on succeeding.
        """
        self.num_harmonizations += 1
        result = next((r for r in node.analysis_result.undecided_results() if any(len(options) > 1 for options in r.primary_selection)), None)
        if result is None:
            return
        parameter_selection = result.primary_selection
        harmonizing_parameter = [parameter for parameter, options in enumerate(parameter_selection) if len(options) > 1][0]
        selection_1 = parameter_selection.copy()
        selection_1[harmonizing_parameter] = [selection_1[harmonizing_parameter][0]]
        selection_2 = parameter_selection.copy()
        selection_2[harmonizing_parameter] = [selection_2[harmonizing_parameter][1]]
        for selection in [selection_1, selection_2]:
            self.verify_parameter_selection(node, selection)

    def verify_parameter_space(self, node):
        self.num_parameter_spaces_considered += 1
        parent_selected_choices = node.parent_info.selected_choices if node.parent_info is not None else None
        node.mdp, node.selected_choices = self.colored_mdp.build(node.parameter_space, parent_selected_choices)

        self.stat.iteration(node.mdp)
        # scheduler_choices is only ever populated by ColoredMdp.scheduler_is_consistent's "dt" branch when the
        # specification is single-property (see split_undecided_space below) -- for a multi-property specification
        # it stays None on every node, so the "scheduler preserved" shortcut must be skipped rather than
        # assumed available, falling through to a real (slower, but correct) model-check instead.
        if node.parent_info is not None and node.parent_info.scheduler_choices is not None:
            for choice in node.parent_info.scheduler_choices:
                if not node.selected_choices[choice]:
                    break
            else:
                # scheduler preserved in the sub-parameter-space
                self.num_schedulers_preserved += 1
                node.analysis_result = node.parent_info.analysis_result
                node.scheduler_choices = node.parent_info.scheduler_choices
                consistent, parameter_selection = self.colored_mdp.are_choices_consistent(node.scheduler_choices, node.parameter_space)
                assert not consistent
                if node.analysis_result.optimality_result is None:
                    for constraint_res in node.analysis_result.constraints_result.results:
                        constraint_res.primary_selection = parameter_selection
                else:
                    node.analysis_result.optimality_result.primary_selection = parameter_selection
                return

        self.num_parameter_spaces_model_checked += 1
        self.check_specification(node)
        if not node.analysis_result.can_improve:
            return
        self.harmonize_inconsistent_scheduler(node)

    def build_unsat_result(self):
        spec_result = paynt.specification.property_result.MdpSpecificationResult()
        spec_result.constraints_result = paynt.specification.property_result.ConstraintsResult([])
        spec_result.optimality_result = paynt.specification.property_result.MdpOptimalityResult(None)
        spec_result.evaluate(None)
        spec_result.can_improve = False
        return spec_result

    def scheduler_scores(self, selection):
        info = self.colored_mdp.feature_info
        inconsistent_assignments = {parameter: options for parameter, options in enumerate(selection) if len(options) > 1}
        assert len(inconsistent_assignments) > 0, f"obtained selection with no inconsistencies: {selection}"
        inconsistent_action_parameters = [
            (parameter, options) for parameter, options in inconsistent_assignments.items() if info.is_action_parameter[parameter]
        ]
        inconsistent_decision_parameters = [
            (parameter, options) for parameter, options in inconsistent_assignments.items() if info.is_decision_parameter[parameter]
        ]
        inconsistent_variable_parameters = [
            (parameter, options) for parameter, options in inconsistent_assignments.items() if info.is_variable_parameter[parameter]
        ]

        # choose one splitter
        splitter = None
        # try action or decision parameters first
        if len(inconsistent_action_parameters) > 0:
            splitter = inconsistent_action_parameters[0][0]
        elif len(inconsistent_decision_parameters) > 0:
            splitter = inconsistent_decision_parameters[0][0]
        else:
            splitter = inconsistent_variable_parameters[0][0]
        assert splitter is not None, "splitter not set"
        # force the score of the selected splitter
        return {splitter: 10}

    def split_undecided_space(self, node):
        mdp = node.mdp
        assert not mdp.is_deterministic

        results = node.analysis_result.undecided_results()

        # (i) L(h): disagreement among the undecided properties (the objective included) whose OWN scheduler is already fully consistent -- same priority as
        # the generic AR path (paper https://www.jair.org/index.php/jair/article/view/16593 Section 3.3)
        candidates = [result.primary_selection for result in results if all(len(options) <= 1 for options in result.primary_selection)]
        disagreements = paynt.utils.scoring.compute_incompatibility_levels(candidates)

        info = self.colored_mdp.feature_info
        if disagreements:
            splitter = paynt.utils.scoring.parameters_with_max_incompatibility(disagreements)[0]
            used_options = disagreements[splitter]
        elif (result := next((r for r in results if any(len(options) > 1 for options in r.primary_selection)), None)) is not None:
            # (ii) DT's own kind-based heuristic (scheduler_scores), applied to the first still-relevant
            # property that actually has a local inconsistency to split on
            parameter_assignments = result.primary_selection
            scores = self.scheduler_scores(parameter_assignments)
            splitter = paynt.utils.scoring.parameters_with_max_score(scores)[0]
            used_options = parameter_assignments[splitter]
        else:
            # (iii) every scheduler left is one tree, which does not settle the node (it violates a constraint, or its value is not valid): set it apart on
            # an action or a decision it uses, else halve a range
            splittable = [parameter for parameter in range(node.parameter_space.num_parameters) if node.parameter_space.parameter_num_options(parameter) > 1]
            used = [parameter for parameter in splittable if len(candidates[0][parameter]) == 1]
            splitter = next((parameter for parameter in used if not info.is_variable_parameter[parameter]), None)
            if splitter is not None:
                [option] = candidates[0][splitter]
                # DFS takes the last one first
                suboptions = [[other for other in node.parameter_space.parameter_options(splitter) if other != option], [option]]
            else:
                splitter = (used or splittable)[0]
                suboptions = node.parameter_space.suboptions_half(splitter)
            return self.split_with_parent_info(node, splitter, suboptions)

        if info.is_action_parameter[splitter] or info.is_decision_parameter[splitter]:
            assert len(used_options) > 1
            core_suboptions, other_suboptions = node.parameter_space.suboptions_enumerate(splitter, used_options)
        else:
            # Variable-kind parameters encode an ordered threshold comparison (e.g. "yellow <= k"), so only a
            # contiguous sub-range of the domain is a valid tree-branch predicate -- unlike action/decision
            # parameters (arbitrary discrete choices), which suboptions_enumerate handles above. Cut the
            # domain just before each disagreeing value's position; for exactly 2 values this is the same
            # single cut as before, and the whole domain is always covered with no residual bucket.
            splitter_options = node.parameter_space.parameter_options(splitter)
            cut_positions = sorted(splitter_options.index(option) for option in used_options)[1:]
            boundaries = [0] + cut_positions + [len(splitter_options)]
            core_suboptions = [splitter_options[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
            for options in core_suboptions:
                assert len(options) > 0
            other_suboptions = []

        if len(other_suboptions) == 0:
            suboptions = core_suboptions
        else:
            suboptions = [other_suboptions] + core_suboptions  # DFS solves core first

        return self.split_with_parent_info(node, splitter, suboptions, node.scheduler_choices)

    def split_with_parent_info(self, node, splitter, suboptions, scheduler_choices=None):
        """The child search nodes of a split.

        :param scheduler_choices: the inconsistent scheduler of a single-property specification, whose result a child keeps while the scheduler stays
            available in it (see verify_parameter_space); None to check every child anew
        """
        # construct corresponding child search nodes (SearchNode.split snapshots node's ParentInfo-relevant
        # state and hands it, shared, to every freshly-split child); layer the DT-specific parent_info
        # fields on top -- these are now always real declared fields (ParentInfo.analysis_result/
        # scheduler_choices), never a dynamic bolt-on that could silently be absent
        child_nodes = node.split(splitter, suboptions)
        assert node.parameter_space.size == sum([child.parameter_space.size for child in child_nodes])
        for child in child_nodes:
            child.parent_info.analysis_result = node.analysis_result
            child.parent_info.scheduler_choices = scheduler_choices
        return child_nodes

    def counters_reset(self):
        self.num_parameter_spaces_considered = 0
        self.num_parameter_spaces_skipped = 0
        self.num_parameter_spaces_model_checked = 0
        self.num_schedulers_preserved = 0
        self.num_harmonizations = 0
        self.num_harmonization_succeeded = 0

    # TODO remove, debugging only
    def counters_print(self):
        logger.info(f"families considered: {self.num_parameter_spaces_considered}")
        logger.info(f"families skipped by construction: {self.num_parameter_spaces_skipped}")
        logger.info(f"families with schedulers preserved: {self.num_schedulers_preserved}")
        logger.info(f"families model checked: {self.num_parameter_spaces_model_checked}")
        logger.info(f"harmonizations attempted: {self.num_harmonizations}")
        logger.info(f"harmonizations succeeded: {self.num_harmonization_succeeded}")
