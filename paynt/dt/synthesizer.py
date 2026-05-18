import paynt.synthesizer.synthesizer_ar
import paynt.dt.factory
import paynt.utils.timer

import paynt.dt.result

from ._utils import simplify_tree

import stormpy
import payntbind

import os
import json

import logging
logger = logging.getLogger(__name__)


def _choose_solver_for_dt_task(paynt_task_dt):
    if paynt_task_dt.has_scheduler_to_map:
        return "dtmap"
    return "dtpaynt"


def _run_dt_map_scheduler(cmdp_factory_dt, scheduler, tree_depth):
    """Helper function to map a scheduler to a decision tree using the DTMap algorithm. Returns a tuple (success, decision_tree)."""

    state_to_choice = payntbind.synthesis.schedulerToStateToGlobalChoice(scheduler, cmdp_factory_dt.quotient_mdp, [x for x in range(cmdp_factory_dt.quotient_mdp.nr_choices)])
    state_to_choice = cmdp_factory_dt.discard_unreachable_choices(state_to_choice)
    choices = cmdp_factory_dt.state_to_choice_to_choices(state_to_choice)

    dt_synthesizer = DtSynthesizer(cmdp_factory_dt)
    dt_synthesizer.map_scheduler(choices, tree_depth=tree_depth)

    simplify_tree(dt_synthesizer.best_tree, cmdp_factory_dt)

    return paynt.dt.result.DtResult(
        success = dt_synthesizer.best_tree is not None,
        value = dt_synthesizer.best_tree_value,
        tree = dt_synthesizer.best_tree
    )


def _run_dtpaynt(cmdp_factory_dt, tree_depth):
    dt_synthesizer = DtSynthesizer(cmdp_factory_dt)
    dt_synthesizer.synthesize_tree(tree_depth)

    simplify_tree(dt_synthesizer.best_tree, cmdp_factory_dt)

    return paynt.dt.result.DtResult(
        success = dt_synthesizer.best_tree is not None,
        value = dt_synthesizer.best_tree_value,
        tree = dt_synthesizer.best_tree
    )



class DtSynthesizer(paynt.synthesizer.synthesizer_ar.SynthesizerAR):

    # tree depth
    tree_depth = 0
    # if set, all trees of size at most tree_depth will be enumerated
    tree_enumeration = False
    # path to a scheduler to be mapped to a decision tree
    scheduler_path = None

    def __init__(self, *args):
        super().__init__(*args)
        self.best_tree = None
        self.best_tree_value = None

    @property
    def method_name(self):
        return "AR (decision tree)"

    def verify_hole_selection(self, family, hole_selection):
        spec = self.quotient.specification
        assignment = family.assume_options_copy(hole_selection)
        dtmc = self.quotient.build_assignment(assignment)
        res = dtmc.check_specification(spec)
        if not res.constraints_result.sat:
            return
        if not spec.has_optimality:
            family.analysis_result.improving_assignment = assignment
            family.analysis_result.can_improve = False
            return
        assignment_value = res.optimality_result.value
        if spec.optimality.improves_optimum(assignment_value):
            # logger.info(f"harmonization achieved value {res.optimality_result.value}")
            self.num_harmonization_succeeded += 1
            family.analysis_result.improving_assignment = assignment
            family.analysis_result.improving_value = assignment_value
            family.analysis_result.can_improve = True
            self.update_optimum(family)


    def harmonize_inconsistent_scheduler(self, family):
        self.num_harmonizations += 1
        mdp = family.mdp
        result = family.analysis_result.undecided_result()
        hole_selection = result.primary_selection
        harmonizing_hole = [hole for hole,options in enumerate(hole_selection) if len(options)>1][0]
        selection_1 = hole_selection.copy(); selection_1[harmonizing_hole] = [selection_1[harmonizing_hole][0]]
        selection_2 = hole_selection.copy(); selection_2[harmonizing_hole] = [selection_2[harmonizing_hole][1]]
        for selection in [selection_1,selection_2]:
            self.verify_hole_selection(family,selection)


    def verify_family(self, family):
        self.num_families_considered += 1
        self.quotient.build(family)

        self.stat.iteration(family.mdp)
        if family.parent_info is not None:
            for choice in family.parent_info.scheduler_choices:
                if not family.selected_choices[choice]:
                    break
            else:
                # scheduler preserved in the sub-family
                self.num_schedulers_preserved += 1
                family.analysis_result = family.parent_info.analysis_result
                family.scheduler_choices = family.parent_info.scheduler_choices
                consistent,hole_selection = self.quotient.are_choices_consistent(family.scheduler_choices, family)
                assert not consistent
                family.analysis_result.optimality_result.primary_selection = hole_selection
                return

        self.num_families_model_checked += 1
        self.check_specification(family)
        if not family.analysis_result.can_improve:
            return
        self.harmonize_inconsistent_scheduler(family)

    def compute_normalized_value(self, value, opt, random):
        return (value-random)/(opt-random) if opt-random != 0 else 1.0

    def counters_reset(self):
        self.num_families_considered = 0
        self.num_families_skipped = 0
        self.num_families_model_checked = 0
        self.num_schedulers_preserved = 0
        self.num_harmonizations = 0
        self.num_harmonization_succeeded = 0

    def counters_print(self):
        logger.info(f"families considered: {self.num_families_considered}")
        logger.info(f"families skipped by construction: {self.num_families_skipped}")
        logger.info(f"families with schedulers preserved: {self.num_schedulers_preserved}")
        logger.info(f"families model checked: {self.num_families_model_checked}")
        logger.info(f"harmonizations attempted: {self.num_harmonizations}")
        logger.info(f"harmonizations succeeded: {self.num_harmonization_succeeded}")
        print()

    def export_decision_tree(self, decision_tree, export_filename_base):
        tree = decision_tree.to_graphviz()
        tree_filename = export_filename_base + ".dot"
        directory = os.path.dirname(tree_filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(tree_filename, 'w') as file:
            file.write(tree.source)
        logger.info(f"exported decision tree to {tree_filename}")

        tree_visualization_filename = export_filename_base + ".png"
        tree.render(export_filename_base, format="png", cleanup=True) # using export_filename_base since graphviz appends .png by default
        logger.info(f"exported decision tree visualization to {tree_visualization_filename}")

        tree_string_filename = export_filename_base + ".txt"
        with open(tree_string_filename, 'w') as file:
            file.write(decision_tree.to_string())
        logger.info(f"exported decision tree string to {tree_string_filename}")


    def synthesize_tree(self, depth : int, family=None):
        self.counters_reset()
        self.quotient.reset_tree(depth)
        self.best_assignment = self.best_assignment_value = None
        self.synthesize(keep_optimum=True)
        if self.best_assignment is not None:
            self.quotient.decision_tree.root.associate_assignment(self.best_assignment)
            self.best_tree = self.quotient.decision_tree
            self.best_tree_value = self.best_assignment_value
        self.best_assignment = self.best_assignment_value = None
        self.counters_print()

    def synthesize_tree_sequence(self, opt_result_value, overall_timeout=None, max_depth=None, break_if_found=False):
        self.best_tree = self.best_tree_value = None

        if max_depth is None:
            max_depth = DtSynthesizer.tree_depth+1
        if overall_timeout is None:
            global_timeout = paynt.utils.timer.GlobalTimer.global_timer.time_limit_seconds
            if global_timeout is None: global_timeout = 900 # TODO this should probably not be the deafult behaviour, we want to run the synthesis indefinitely if the user does not give us timeout
            overall_timeout = global_timeout
            tree_sequence_timer = None
        else:
            tree_sequence_timer = paynt.utils.timer.Timer(overall_timeout)
            tree_sequence_timer.start()
        depth_timeout = overall_timeout / 2 / (max_depth-1) if max_depth > 1 else overall_timeout
        for depth in range(max_depth):
            print()
            self.quotient.reset_tree(depth)
            best_assignment_old = self.best_assignment

            family = self.quotient.family
            self.explored = 0
            self.counters_reset()
            self.stat = paynt.synthesizer.statistic.Statistic(self)
            self.stat.start(family)
            timeout = depth_timeout if depth < max_depth-1 else overall_timeout / 2 # second half of the time for the last depth
            self.synthesis_timer = paynt.utils.timer.Timer(timeout)
            self.synthesis_timer.start()
            families = [family]

            if self.best_tree is not None:
                subfamily = family.copy()
                self.quotient.decision_tree.root.apply_hint(subfamily,self.best_tree.root)
                families = [subfamily,family]

            for family in families:
                self.synthesize_one(family)
            self.stat.finished_synthesis()
            self.stat.print()
            self.synthesis_timer = None
            self.counters_print()

            new_assignment_synthesized = self.best_assignment != best_assignment_old
            if new_assignment_synthesized:
                logger.info("printing synthesized assignment below:")
                logger.info(self.best_assignment)

                if self.best_assignment is not None and self.best_assignment.size == 1:
                    dtmc = self.quotient.build_assignment(self.best_assignment)
                    result = dtmc.check_specification(self.quotient.specification)
                    logger.info(f"double-checking specification satisfiability: {result}")

                self.best_tree = self.quotient.decision_tree
                self.best_tree.root.associate_assignment(self.best_assignment)
                self.best_tree_value = self.best_assignment_value

                if break_if_found or (opt_result_value != 0 and abs( (self.best_assignment_value-opt_result_value)/opt_result_value ) < 1e-3) or (opt_result_value == 0 and self.best_assignment_value < 1e-3):
                    break

            if self.resource_limit_reached() or tree_sequence_timer is not None and tree_sequence_timer.time_limit_reached():
                break

    def map_scheduler(self, scheduler_choices, tree_depth=None):
        self.counters_reset()
        if tree_depth is None:
            tree_depth = DtSynthesizer.tree_depth
        for depth in range(tree_depth+1):
            self.quotient.reset_tree(depth,enable_harmonization=False)
            family = self.quotient.family.copy()
            family.analysis_result = self.quotient.build_unsat_result()
            self.quotient.build(family)
            consistent,hole_selection = self.quotient.are_choices_consistent(scheduler_choices, family)
            if consistent:
                self.verify_hole_selection(family,hole_selection)
                if self.best_assignment is not None:
                    self.best_tree = self.quotient.decision_tree
                    self.best_tree.root.associate_assignment(self.best_assignment)
                    self.best_tree_value = self.best_assignment_value
                    break

            if self.resource_limit_reached():
                break

    def run(self, optimum_threshold=None):
        # self.quotient.reset_tree(DtSynthesizer.tree_depth,enable_harmonization=True)
        scheduler_choices = None
        if DtSynthesizer.scheduler_path is None:
            paynt_mdp = paynt.models.models.Mdp(self.quotient.quotient_mdp)
            mc_result = paynt_mdp.model_check_property(self.quotient.get_property())
        else:
            opt_result_value = None
            with open(DtSynthesizer.scheduler_path, 'r') as f:
                scheduler_json = json.load(f)
            scheduler_choices,scheduler_json_relevant = self.quotient.scheduler_json_to_choices(scheduler_json, discard_unreachable_states=True)

            # export transformed scheduler
            # import os
            # directory = os.path.dirname(DtSynthesizer.scheduler_path)
            # transformed_name = f"scheduler-reachable.storm.json"
            # scheduler_relevant_path = os.path.join(directory, transformed_name)
            # with open(scheduler_relevant_path, 'w') as f:
            #     json.dump(scheduler_json_relevant, f, indent=4)
            # logger.debug(f"stored transformed scheduler to {scheduler_relevant_path}")
            # exit()

            submdp = self.quotient.build_from_choice_mask(scheduler_choices)
            mc_result = submdp.model_check_property(self.quotient.get_property())
        opt_result_value = mc_result.value
        logger.info(f"the optimal scheduler has value: {opt_result_value}")

        if self.quotient.DONT_CARE_ACTION_LABEL in self.quotient.action_labels:
            random_choices = self.quotient.get_random_choices()
            submdp_random = self.quotient.build_from_choice_mask(random_choices)
            mc_result_random = submdp_random.model_check_property(self.quotient.get_property())
            random_result_value = mc_result_random.value
            logger.info(f"the random scheduler has value: {random_result_value}")
            # self.set_optimality_threshold(random_result_value)

        self.best_assignment = self.best_assignment_value = None
        self.best_tree = self.best_tree_value = None
        if scheduler_choices is not None:
            self.map_scheduler(scheduler_choices)
        else:
            if self.quotient.specification.has_optimality:
                epsilon = 1e-1
                mc_result_positive = opt_result_value > 0
                if self.quotient.specification.optimality.maximizing == mc_result_positive:
                    epsilon *= -1
                # optimum_threshold = opt_result_value * (1 + epsilon)
            self.set_optimality_threshold(optimum_threshold)

            if not DtSynthesizer.tree_enumeration:
                self.synthesize_tree(DtSynthesizer.tree_depth)
            else:
                self.synthesize_tree_sequence(opt_result_value)

        logger.info(f"the optimal scheduler has value: {opt_result_value}")
        if self.quotient.DONT_CARE_ACTION_LABEL in self.quotient.action_labels:
            logger.info(f"the random scheduler has value: {random_result_value}")
        if self.best_tree is None:
            logger.info("no admissible tree found")
        else:
            relevant_state_valuations = [self.quotient.relevant_state_valuations[state] for state in self.quotient.state_is_relevant_bv]
            self.best_tree.simplify(relevant_state_valuations)
            depth = self.best_tree.get_depth()
            num_nodes = len(self.best_tree.collect_nonterminals())
            logger.info(f"synthesized tree of depth {depth} with {num_nodes} decision nodes")
            if self.quotient.specification.has_optimality:
                logger.info(f"the synthesized tree has value {self.best_tree_value}")
                if self.quotient.DONT_CARE_ACTION_LABEL in self.quotient.action_labels:
                    logger.info(f"the synthesized tree has relative value: {self.compute_normalized_value(self.best_tree_value, opt_result_value, random_result_value)}")
            logger.info(f"printing the synthesized tree below:")
            print(self.best_tree.to_string())

            if self.export_synthesis_filename_base is not None:
                self.export_decision_tree(self.best_tree, self.export_synthesis_filename_base)

        time_total = round(paynt.utils.timer.GlobalTimer.read(),2)
        logger.info(f"synthesis finished after {time_total} seconds")

        return self.best_tree
