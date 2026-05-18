from ..decision_tree import DecisionTree
from ..factory import DtColoredMdpFactory
from ..synthesizer import DtSynthesizer
import paynt.utils.timer
from ._utils import create_uniform_random_tree, get_submdp_from_unfixed_states, build_tree_helper_tree, get_state_space_for_tree_helper_node, dt_to_scheduler_json, run_dtcontrol, parse_tree_helper_json, run_scikit_learn_tree, state_to_choice_to_state_to_action, dt_to_state_to_actions

import stormpy
import payntbind

import json

import logging
logger = logging.getLogger(__name__)

class DtNest(DtSynthesizer):

    initial_tree_path = None

    def __init__(self, *args):
        super().__init__(*args)
        self.best_tree = None
        self.best_tree_value = None

        # TODO add cli option to set these settings
        self.initialize_settings()

    def initialize_settings(self):
        self.subtree_depth = 7
        self.max_iter = 100000 # max number of subtrees to be investigated
        self.epsilon = 0.05
        self.timeout = 600
        self.depth_fine_tuning = True # decreases sub-tree depth once all subtrees of the current depth have been explored
        self.break_on_small_tree = True # dtPAYNT synthesis ends when an implementable tree with good enough value is found
        self.use_dtcontrol = False # If set to False, scikit-learn is used instead of dtcontrol
        self.allow_perturbations = True # If False all perturbations are disabled, meaning we are only using dtPAYNT to make the initial tree smaller
        self.recompute_scheduler = True # recomputes scheduler for the subtree outside of the replaced subtree
        # dtcontrol_settings = ["default"] # this defines the different settings we run dtcontrol with # TODO this is currently not nicely supported in dtcontrol
        # dtcontrol_settings = ["default", "gini", "entropy", "maxminority"] # other possible settings: gini, entropy, maxminority
        self.use_states_for_node_priority = False # this is super slow for some models but should mean better prioritization

    @property
    def method_name(self):
        return "dtNESt"

    def counters_reset(self):
        # integration stats
        self.dt_learning_calls = 0
        self.dt_learning_successes = 0
        self.dt_learning_recomputed_calls = 0
        self.dt_learning_recomputed_successes = 0
        self.dtpaynt_calls = 0
        self.dtpaynt_successes_smaller = 0
        self.dtpaynt_tree_found = 0
        self.all_larger = 0

    def choose_tree_to_use(self, current_tree, dtpaynt_tree, dtlearn_trees, recomputed_scheduler_trees):
        # this also defines the priority in case of a tie, therefore: current > dtpaynt > dtlearn > recomputed
        current_nodes = len(current_tree.collect_nonterminals())
        nodes = {"current": [current_nodes, current_tree.get_depth(), 1]}
        for setting, dtlearn_tree in recomputed_scheduler_trees.items():
            nodes["recomputed-"+setting] = [len(dtlearn_tree[1].collect_nonterminals()), dtlearn_tree[1].get_depth(), 1]
        for setting, dtlearn_tree in dtlearn_trees.items():
            nodes["dtlearn-"+setting] = [len(dtlearn_tree[1].collect_nonterminals()), dtlearn_tree[1].get_depth(), 1]
        nodes["dtpaynt"] = [len(dtpaynt_tree.collect_nonterminals()), dtpaynt_tree.get_depth(), 1]
        nodes = {k: v for k, v in nodes.items() if v is not None}
        sorted_nodes = sorted(nodes.items(), key=lambda item: item[1][1])
        sorted_nodes = sorted(nodes.items(), key=lambda item: item[1][0])

        logger.info(f"subtree information: {sorted_nodes}")
        return sorted_nodes[0][0]
    

    def create_tree_node_queue_heuristic(self, helper_tree, desired_depth=6, nodes_to_skip=[], use_states_for_node_priority=False):
        nodes = helper_tree.collect_nodes(lambda node : node.get_depth() == desired_depth)
        if nodes is None or len(nodes) == 0:
            return []
        helper_nodes = [self.quotient.tree_helper[node.identifier] for node in nodes]
        helper_node_stats = []
        for helper_node in helper_nodes:
            if helper_node["id"] == 0 or helper_node["id"] in nodes_to_skip:
                continue
            helper_tree_node = helper_tree.collect_nodes(lambda node : node.identifier == helper_node["id"])[0]
            if use_states_for_node_priority:
                stats = {"id": helper_node["id"], "states": get_state_space_for_tree_helper_node(self.quotient, helper_node["id"]), "nodes": helper_tree_node.get_number_of_descendants()}
            else:
                stats = {"id": helper_node["id"], "nodes": helper_tree_node.get_number_of_descendants()}

            # this happens for nodes created outside of DtControl
            if "evaluations" not in helper_node.keys():
                stats["predicates"] = {}
            else:
                stats["predicates"] = {pred : eval for pred, eval in helper_node["evaluations"].items() if eval <= list(helper_node["evaluations"].values())[0]*1.05}
            helper_node_stats.append(stats)

        if len(helper_node_stats) == 0:
            return []

        helper_node_stats = sorted(helper_node_stats, key=lambda x : x["nodes"], reverse=True)
        helper_node_stats = sorted(helper_node_stats, key=lambda x : len(x["predicates"]), reverse=True)
        if use_states_for_node_priority:
            helper_node_stats = sorted(helper_node_stats, key=lambda x : x["states"].number_of_set_bits()/x["nodes"])

        return helper_node_stats
    

    def synthesize_subtrees(self, opt_result_value, random_result_value=None):

        # init
        self.counters_reset()
        if random_result_value is None:
            # if we dont have the random result value just make the threshold to be the epsilon of optimum value
            mc_result_positive = opt_result_value > 0
            if self.quotient.specification.optimality.maximizing == mc_result_positive:
                self.epsilon *= -1
            eps_optimum_threshold = opt_result_value * (1 + self.epsilon)
        else: # this should result in normalised value of the produced tree being within espilon
            opt_random_diff = opt_result_value - random_result_value
            eps_optimum_threshold = opt_result_value - self.epsilon * opt_random_diff
        self.synthesis_timer = paynt.utils.timer.Timer(self.timeout)
        self.synthesis_timer.start()

        # initialize from external tree
        self.quotient.tree_helper_tree = build_tree_helper_tree(self.quotient)
        tree_helper_tree = self.quotient.tree_helper_tree
        logger.info(f'initial external tree has depth {tree_helper_tree.get_depth()} and {len(tree_helper_tree.collect_nonterminals())} nodes')
        
        current_iter = 0
        current_depth = self.subtree_depth

        while (self.depth_fine_tuning and current_depth > 1) or (current_depth == self.subtree_depth):

            if self.synthesis_timer.time_limit_reached():
                    logger.info(f"timeout reached")
                    break

            logger.info(f"starting iteration with subtree depth {current_depth}")
            # TODO this is not guaranteed to work in subsequent iterations when the dtPAYNT tree is used
            # I don't know what this means anymore...
            node_queue = self.create_tree_node_queue_heuristic(tree_helper_tree, desired_depth=current_depth, use_states_for_node_priority=self.use_states_for_node_priority)

            while len(node_queue) > 0 and current_iter < self.max_iter:

                if self.synthesis_timer.time_limit_reached():
                    logger.info(f"timeout reached")
                    break

                logger.info(f"starting iteration {current_iter} with {len(node_queue)} nodes in node queue")
                logger.info(f"current tree size: {len(tree_helper_tree.collect_nonterminals())} decision nodes")

                current_iter += 1
                node = node_queue.pop(0)

                # subtree synthesis
                if self.use_states_for_node_priority:
                    node_states = node["states"]
                else:
                    node_states = get_state_space_for_tree_helper_node(self.quotient, node["id"])
                submdp = get_submdp_from_unfixed_states(self.quotient, node_states)
                logger.info(f"subtree quotient has {submdp.model.nr_states} states and {submdp.model.nr_choices} choices")
                subtree_spec = self.quotient.specification.copy()
                subtree_quotient = DtColoredMdpFactory(submdp.model, subtree_spec)
                subtree_quotient.specification.optimality.update_optimum(eps_optimum_threshold)
                subtree_synthesizer = DtSynthesizer(subtree_quotient)
                self.dtpaynt_calls += 1
                
                if subtree_quotient.state_is_relevant_bv.number_of_set_bits() == 0:
                    random_tree = create_uniform_random_tree(subtree_quotient)
                    subtree_synthesizer.best_tree = random_tree
                else:
                    subtree_synthesizer.synthesize_tree_sequence(opt_result_value, overall_timeout=60, max_depth=current_depth, break_if_found=self.break_on_small_tree)

                # create new tree
                if subtree_synthesizer.best_tree is not None:
                    logger.info(f"admissible subtree found from node {node['id']}")
                    self.dtpaynt_tree_found += 1
                    relevant_state_valuations = [subtree_quotient.relevant_state_valuations[state] for state in subtree_quotient.state_is_relevant_bv]
                    subtree_synthesizer.best_tree.simplify(relevant_state_valuations)
                    dtpaynt_subtree_helper_tree_copy = tree_helper_tree.copy()
                    dtpaynt_subtree_helper_tree_copy.append_tree_as_subtree(subtree_synthesizer.best_tree, node["id"], subtree_quotient)
                    dtpaynt_subtree_helper_tree_copy.root.assign_identifiers(keep_old=True)
                    logger.info(f'new tree has depth {dtpaynt_subtree_helper_tree_copy.get_depth()} and {len(dtpaynt_subtree_helper_tree_copy.collect_nonterminals())} nodes')

                    self.quotient.tree_helper_tree = dtpaynt_subtree_helper_tree_copy

                    new_tree_helper_tree = None
                    recomputed_scheduler_tree_helper_tree = None
                    dtlearn_trees = {}
                    recomputed_dtlearn_trees = {}

                    # exit()

                    if self.allow_perturbations:

                        submdp_for_tree = get_submdp_from_unfixed_states(self.quotient, node_states)
                        reachable_states = stormpy.BitVector(self.quotient.quotient_mdp.nr_states, False)
                        for state in range(submdp_for_tree.model.nr_states):
                            reachable_states.set(submdp_for_tree.quotient_state_map[state], True)

                        if self.recompute_scheduler:
                            submpd_dtlearn_of_subtree = get_submdp_from_unfixed_states(self.quotient, ~node_states)
                            oos_result = submpd_dtlearn_of_subtree.check_specification(self.quotient.specification)
                            new_scheduler = oos_result.optimality_result.result.scheduler
                            state_to_choice = self.quotient.scheduler_to_state_to_choice(submpd_dtlearn_of_subtree, new_scheduler)

                            if self.use_dtcontrol:
                                recomputed_scheduler = payntbind.synthesis.create_scheduler(self.quotient.quotient_mdp.nr_states)
                                quotient_mdp_nci = self.quotient.quotient_mdp.nondeterministic_choice_indices.copy()
                                for state in range(self.quotient.quotient_mdp.nr_states):
                                    quotient_choice = state_to_choice[state]
                                    if quotient_choice is None or not self.quotient.state_is_relevant_bv.get(state):
                                        payntbind.synthesis.set_dont_care_state_for_scheduler(recomputed_scheduler, state, 0, False)
                                        index = 0
                                    else:
                                        index = quotient_choice - quotient_mdp_nci[state]
                                    scheduler_choice = stormpy.storage.SchedulerChoice(index)
                                    recomputed_scheduler.set_choice(scheduler_choice, state)

                                recomputed_json_scheduler_full = json.loads(recomputed_scheduler.to_json_str(self.quotient.quotient_mdp, skip_dont_care_states=True))
                                recomputed_json_str = json.dumps(recomputed_json_scheduler_full, indent=4)

                        # calling dtcontrol
                        # TODO I removed the option to call multiple dtcontrol settings, because they are not nicely implemented in dtcontrol, if they fix that I will reintroduce it
                        if self.use_dtcontrol:
                            scheduler_json = dt_to_scheduler_json(dtpaynt_subtree_helper_tree_copy, self.quotient, reachable_states)

                            new_learned_tree_helper = run_dtcontrol(scheduler_json, "storm.json")
                            self.dt_learning_calls += 1

                            if self.recompute_scheduler:

                                recomputed_scheduler_tree_helper = run_dtcontrol(recomputed_json_str, "storm.json")
                                self.dt_learning_recomputed_calls += 1

                        else:

                            state_to_action = dt_to_state_to_actions(dtpaynt_subtree_helper_tree_copy, self.quotient, reachable_states)
                            new_learned_tree_helper = run_scikit_learn_tree(self.quotient.relevant_state_valuations, state_to_action, self.quotient.variables, self.quotient.action_labels)
                            self.dt_learning_calls += 1

                            if self.recompute_scheduler:

                                recomputed_state_to_action = state_to_choice_to_state_to_action(state_to_choice, self.quotient)
                                recomputed_scheduler_tree_helper = run_scikit_learn_tree(self.quotient.relevant_state_valuations, 
                                recomputed_state_to_action, self.quotient.variables, self.quotient.action_labels)
                                self.dt_learning_recomputed_calls += 1


                        new_tree_helper_tree = build_tree_helper_tree(self.quotient, new_learned_tree_helper)
                        logger.info(f'new learned tree (default) has depth {new_tree_helper_tree.get_depth()} and {len(new_tree_helper_tree.collect_nonterminals())} nodes')
                        dtlearn_trees["default"] = (new_learned_tree_helper, new_tree_helper_tree)

                        if self.recompute_scheduler:
                            recomputed_scheduler_tree_helper_tree = build_tree_helper_tree(self.quotient, recomputed_scheduler_tree_helper)
                            logger.info(f'new learned tree (default) based on recomputed scheduler has depth {recomputed_scheduler_tree_helper_tree.get_depth()} and {len(recomputed_scheduler_tree_helper_tree.collect_nonterminals())} nodes')

                            recomputed_dtlearn_trees["default"] = (recomputed_scheduler_tree_helper, recomputed_scheduler_tree_helper_tree)



                    chosen_tree = self.choose_tree_to_use(tree_helper_tree, dtpaynt_subtree_helper_tree_copy, dtlearn_trees, recomputed_dtlearn_trees)

                    if chosen_tree == "current":
                        logger.info(f"None of the new trees are smaller, continuing with current tree")
                        self.all_larger += 1
                        self.quotient.tree_helper_tree = tree_helper_tree

                    elif chosen_tree == "dtpaynt":
                        logger.info(f"New dtPAYNT tree is smallest")
                        self.dtpaynt_successes_smaller += 1
                        tree_helper_tree = dtpaynt_subtree_helper_tree_copy
                        self.quotient.tree_helper_tree = tree_helper_tree
                        for node in node_queue:
                            nodes = self.quotient.tree_helper_tree.collect_nodes(lambda x : x.old_identifier == node["id"])
                            assert len(nodes) == 1, f'only one node should have the old_identifier equal to {node["id"]}'
                            new_node = nodes[0]
                            node["id"] = new_node.identifier
                        new_nodes = self.create_tree_node_queue_heuristic(tree_helper_tree, desired_depth=current_depth, nodes_to_skip=[node["id"] for node in node_queue], use_states_for_node_priority=self.use_states_for_node_priority)
                        node_queue += new_nodes

                    elif chosen_tree.startswith("dtlearn"):
                        logger.info(f"New learned tree ({chosen_tree}) is smallest")
                        dtlearn_setting = chosen_tree.split("-")[1]
                        new_dtlearn_tree_helper = dtlearn_trees[dtlearn_setting][0]
                        new_dtlearn_tree_helper_tree = dtlearn_trees[dtlearn_setting][1]
                        self.dt_learning_successes += 1
                        self.quotient.tree_helper = new_dtlearn_tree_helper
                        self.quotient.tree_helper_tree = new_dtlearn_tree_helper_tree
                        tree_helper_tree = new_dtlearn_tree_helper_tree
                        node_queue = self.create_tree_node_queue_heuristic(tree_helper_tree, use_states_for_node_priority=self.use_states_for_node_priority)

                    elif chosen_tree.startswith("recomputed"):
                        logger.info(f"New learned tree ({chosen_tree}) for recomputed scheduler is smallest")
                        dtlearn_setting = chosen_tree.split("-")[1]
                        recomputed_scheduler_tree_helper = recomputed_dtlearn_trees[dtlearn_setting][0]
                        recomputed_scheduler_tree_helper_tree = recomputed_dtlearn_trees[dtlearn_setting][1]
                        self.dt_learning_recomputed_successes += 1
                        self.quotient.tree_helper = recomputed_scheduler_tree_helper
                        self.quotient.tree_helper_tree = recomputed_scheduler_tree_helper_tree
                        tree_helper_tree = recomputed_scheduler_tree_helper_tree
                        node_queue = self.create_tree_node_queue_heuristic(tree_helper_tree, use_states_for_node_priority=self.use_states_for_node_priority)
                    
                else:
                    logger.info(f"no admissible subtree found from node {node['id']}")

            current_depth -= 1

        self.quotient.tree_helper_tree = tree_helper_tree
        
        self.synthesis_timer.stop()

        # double check
        dtmc = get_submdp_from_unfixed_states(self.quotient)
        # TODO why would it not be a DTMC here???
        if dtmc.model.nr_states != dtmc.model.nr_choices:
            logger.info(f"tree did not induce dtmc?")
        # assert dtmc.model.nr_states == dtmc.model.nr_choices, "tree did not induce dtmc"
        result = dtmc.check_specification(self.quotient.specification)

        # TODO find out why this would not hold????
        if opt_result_value < eps_optimum_threshold:
            assert result.optimality_result.value <= eps_optimum_threshold, f"optimum value {result.optimality_result.value} is not below threshold {eps_optimum_threshold}"
        else:
            assert result.optimality_result.value >= eps_optimum_threshold, f"optimum value {result.optimality_result.value} is not above threshold {eps_optimum_threshold}"

        self.best_tree = self.quotient.tree_helper_tree
        self.best_tree_value = result.optimality_result.value

        logger.info(f'final tree has value {result.optimality_result.value} with depth {self.quotient.tree_helper_tree.get_depth()} and {len(self.quotient.tree_helper_tree.collect_nonterminals())} nodes')


    def run(self, optimum_threshold=None):

        paynt_mdp = paynt.models.models.SubMdp(self.quotient.quotient_mdp, [x for x in range(self.quotient.quotient_mdp.nr_states)], [x for x in range(self.quotient.quotient_mdp.nr_choices)])
        if len(self.quotient.specification.constraints) > 0:
            from ._utils import get_optimality_specification
            self.quotient.specification = get_optimality_specification(self.quotient.specification)
        mc_result = paynt_mdp.model_check_property(self.quotient.get_property())
        opt_scheduler = mc_result.result.scheduler

        state_to_choice = self.quotient.scheduler_to_state_to_choice(paynt_mdp, opt_scheduler)

        if self.initial_tree_path is None:

            if self.use_dtcontrol:
        
                # creating the initial scheduler for dtcontrol
                relevant_opt_scheduler = payntbind.synthesis.create_scheduler(self.quotient.quotient_mdp.nr_states)
                nci = self.quotient.quotient_mdp.nondeterministic_choice_indices.copy()
                for state in range(self.quotient.quotient_mdp.nr_states):
                    quotient_choice = state_to_choice[state]
                    if quotient_choice is None or not self.quotient.state_is_relevant_bv.get(state):
                        payntbind.synthesis.set_dont_care_state_for_scheduler(relevant_opt_scheduler, state, 0, False)
                        index = 0
                    else:
                        index = quotient_choice - nci[state]
                    scheduler_choice = stormpy.storage.SchedulerChoice(index)
                    relevant_opt_scheduler.set_choice(scheduler_choice, state)

                relevant_opt_scheduler_full = json.loads(relevant_opt_scheduler.to_json_str(self.quotient.quotient_mdp, skip_dont_care_states=True))
                relevant_opt_scheduler_str = json.dumps(relevant_opt_scheduler_full, indent=4)

                initial_tree_helper = run_dtcontrol(relevant_opt_scheduler_str, "storm.json")

            else: # using scikit-learn to obtain the initial tree helper

                state_to_action = state_to_choice_to_state_to_action(state_to_choice, self.quotient)
                initial_tree_helper = run_scikit_learn_tree(self.quotient.relevant_state_valuations, state_to_action, self.quotient.variables, self.quotient.action_labels)

            
        else:

            logger.info(f"parsing initial tree from {self.initial_tree_path}")
            initial_tree_helper = parse_tree_helper_json(self.initial_tree_path)

        self.quotient.tree_helper = initial_tree_helper

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

        assert self.quotient.tree_helper is not None, "tree helper not set, cannot run dtNest"

        self.synthesize_subtrees(opt_result_value, random_result_value)

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

            # integration logs
            if self.quotient.tree_helper is not None:
                logger.info(f"dt learning calls: {self.dt_learning_calls}")
                logger.info(f"dt learning successes: {self.dt_learning_successes}")
                logger.info(f"dt learning recomputed calls: {self.dt_learning_recomputed_calls}")
                logger.info(f"dt learning recomputed successes: {self.dt_learning_recomputed_successes}")
                logger.info(f"dtpaynt calls: {self.dtpaynt_calls}")
                logger.info(f"dtpaynt successes smaller: {self.dtpaynt_successes_smaller}")
                logger.info(f"dtpaynt tree found: {self.dtpaynt_tree_found}")
                logger.info(f"all larger: {self.all_larger}")

            if self.export_synthesis_filename_base is not None:
                self.export_decision_tree(self.best_tree, self.export_synthesis_filename_base)

        time_total = round(paynt.utils.timer.GlobalTimer.read(),2)
        logger.info(f"synthesis finished after {time_total} seconds")

        print()
        print(self.best_tree.to_string())

        return self.best_tree
