import payntbind
import stormpy

from ..decision_tree import DecisionTree

import json

import os
from math import floor
import shutil
import subprocess
from datetime import datetime

from sklearn import tree

import logging
logger = logging.getLogger(__name__)


def dt_to_scheduler_json(decision_tree, quotient, reachable_states=None):
    if reachable_states is None:
        reachable_states = stormpy.BitVector(quotient.quotient_mdp.nr_states, True)
    scheduler = payntbind.synthesis.create_scheduler(quotient.quotient_mdp.nr_states)
    nci = quotient.quotient_mdp.nondeterministic_choice_indices.copy()
    for state in range(quotient.quotient_mdp.nr_states):
        if quotient.state_is_relevant_bv.get(state) and reachable_states.get(state):
            action_index = get_action_for_state(decision_tree.root, quotient, state, quotient.relevant_state_valuations[state], nci)
        else:
            # this should leave undefined in the scheduler and will be filtered below
            payntbind.synthesis.set_dont_care_state_for_scheduler(scheduler, state, 0, False) # trying to filter out these states through Storm
            continue
        scheduler_choice = stormpy.storage.SchedulerChoice(action_index)
        scheduler.set_choice(scheduler_choice, state)
        
    json_scheduler_full = json.loads(scheduler.to_json_str(quotient.quotient_mdp, skip_dont_care_states=True))

    json_str = json.dumps(json_scheduler_full, indent=4)
    return json_str

def dt_to_state_to_actions(decision_tree, quotient, reachable_states=None):
    if reachable_states is None:
        reachable_states = stormpy.BitVector(quotient.quotient_mdp.nr_states, True)
    state_to_action = []
    nci = quotient.quotient_mdp.nondeterministic_choice_indices.copy()
    for state in range(quotient.quotient_mdp.nr_states):
        if quotient.state_is_relevant_bv.get(state) and reachable_states.get(state):
            action_index = get_action_for_state(decision_tree.root, quotient, state, quotient.relevant_state_valuations[state], nci)
            state_to_action.append(quotient.choice_to_action[nci[state] + action_index])
        else:
            state_to_action.append(-1)

    return state_to_action


def get_action_for_state(node, quotient, state, state_valuation, nci):
    if node.is_terminal:
        action_index = node.action
        index = 0
        for choice in range(nci[state],nci[state+1]):
            if quotient.choice_to_action[choice] == action_index:
                return index
            index += 1
        else:
            # TODO as far as I know this happens only because of unreachable states not being included in the tree
            # for now we will treat this by using the __random__ action but it can lead to strange behaviour
            index = 0
            for choice in range(nci[state],nci[state+1]):
                if quotient.action_labels[quotient.choice_to_action[choice]] == "__random__":
                    return index
                index += 1
            assert False
    var = quotient.variables[node.variable]
    bound = var.domain[node.variable_bound]
    if state_valuation[node.variable] <= bound:
        return get_action_for_state(node.child_true, quotient, state, state_valuation, nci)
    else:
        return get_action_for_state(node.child_false, quotient, state, state_valuation, nci)


def get_states_satisfying_predicate(dt_colored_mdp_factory, node, current_states, leq=True):
        bound = dt_colored_mdp_factory.variables[node.variable].domain[node.variable_bound]
        for state,state_valuation in enumerate(dt_colored_mdp_factory.relevant_state_valuations):
            if not current_states.get(state):
                continue
            if leq and state_valuation[node.variable] > bound:
                current_states.set(state, False)
            elif not leq and state_valuation[node.variable] <= bound:
                current_states.set(state, False)
        return current_states

def get_state_space_for_tree_helper_node(dt_colored_mdp_factory, node_id):
    node = dt_colored_mdp_factory.tree_helper_tree.collect_nodes(lambda node : node.identifier == node_id)[0]
    current_node = node
    states = stormpy.storage.BitVector(dt_colored_mdp_factory.quotient_mdp.nr_states, True)
    while current_node.parent is not None:
        parent_node = current_node.parent
        if parent_node.child_true.identifier == current_node.identifier:
            states = get_states_satisfying_predicate(dt_colored_mdp_factory, parent_node, states, leq=True)
        else:
            states = get_states_satisfying_predicate(dt_colored_mdp_factory, parent_node, states, leq=False)
        current_node = parent_node
    return states

def get_chosen_action_for_state_from_tree_helper(dt_colored_mdp_factory, state, tree):
    state_valuation = dt_colored_mdp_factory.relevant_state_valuations[state]
    current_node = tree.root
    while not current_node.is_terminal:
        bound = dt_colored_mdp_factory.variables[current_node.variable].domain[current_node.variable_bound]
        if state_valuation[current_node.variable] <= bound:
            current_node = current_node.child_true
        else:
            current_node = current_node.child_false
    return dt_colored_mdp_factory.action_labels[current_node.action]

def get_selected_choices_from_tree_helper(dt_colored_mdp_factory, state_to_exclude, tree=None):
    if tree is None:
        tree = dt_colored_mdp_factory.tree_helper_tree
    selected_choices = stormpy.storage.BitVector(dt_colored_mdp_factory.quotient_mdp.nr_choices, False)
    mdp_nci = dt_colored_mdp_factory.quotient_mdp.nondeterministic_choice_indices.copy()
    for state in range(dt_colored_mdp_factory.quotient_mdp.nr_states):
        if state_to_exclude.get(state) or dt_colored_mdp_factory.state_is_relevant_bv.get(state) == False:
            for choice in range(mdp_nci[state],mdp_nci[state+1]):
                selected_choices.set(choice, True)
            continue
        chosen_action_label = get_chosen_action_for_state_from_tree_helper(dt_colored_mdp_factory, state, tree)
        action_index = dt_colored_mdp_factory.action_labels.index(chosen_action_label)
        for choice in range(mdp_nci[state],mdp_nci[state+1]):
            if dt_colored_mdp_factory.choice_to_action[choice] == action_index:
                selected_choices.set(choice, True)
                break
        else:
            # TODO as far as I know this happens only because of unreachable states not being included in the tree
            # for now we will treat this by using the __random__ action but it can lead to strange behaviour
            for choice in range(mdp_nci[state],mdp_nci[state+1]):
                if dt_colored_mdp_factory.action_labels[dt_colored_mdp_factory.choice_to_action[choice]] == "__random__":
                    selected_choices.set(choice, True)
                    break
            continue
            assert False, f"no choice for state {state} even though action {chosen_action_label} was chosen"

    return selected_choices


def build_tree_helper_tree(dt_colored_mdp_factory, tree_helper=None):
    if tree_helper is None:
        tree_helper = dt_colored_mdp_factory.tree_helper
    helper_tree = DecisionTree(dt_colored_mdp_factory.action_labels,dt_colored_mdp_factory.variables)
    helper_tree.build_from_tree_helper(tree_helper)
    return helper_tree

# unfixed_states is a bitvector of states that should be left unfixed in the submdp
def get_submdp_from_unfixed_states(dt_colored_mdp_factory, unfixed_states=None):
    if unfixed_states is None:
        unfixed_states = stormpy.storage.BitVector(dt_colored_mdp_factory.quotient_mdp.nr_states, False)
    selected_choices = get_selected_choices_from_tree_helper(dt_colored_mdp_factory, unfixed_states)
    submdp = dt_colored_mdp_factory.build_from_choice_mask(selected_choices)
    return submdp

# in dtNest, there might be an MDP for a subtree with no relevant states, in that case we want to replace this subtree with a random action
def create_uniform_random_tree(dt_colored_mdp_factory):
    decision_tree = DecisionTree(dt_colored_mdp_factory.action_labels,dt_colored_mdp_factory.variables)
    decision_tree.random_tree()
    return decision_tree


def build_tree_helper(tree_node_json, helper=[]):
    current_index = len(helper)
    if tree_node_json['split'] is None:
        # TODO this is a temp fix that only works for some models...
        helper.append({'id': current_index, 'leaf': True, 'chosen': tree_node_json['actual_label']})
        return helper
    helper.append({'id': current_index, 'leaf': False, 'chosen': (tree_node_json['split']['lhs']['var'], floor(tree_node_json['split']['rhs'])), 'children': [], 'evaluations': {(x['split']['lhs']['var'], floor(x['split']['rhs'])): x['impurity'] for x in tree_node_json['additional_splits']}})
    # sort the evaluations by impurity value
    helper[current_index]['evaluations'] = {k: v for k, v in sorted(helper[current_index]['evaluations'].items(), key=lambda item: item[1])}

    assert len(tree_node_json['children']) == 2, "expected two children"
    # left child
    assert tree_node_json['children'][0]['edge_label'] == "true", "expected left child edge label to be True"
    helper[current_index]['children'].append(len(helper))
    helper = build_tree_helper(tree_node_json['children'][0], helper)
    # right child
    assert tree_node_json['children'][1]['edge_label'] == "false", "expected right child edge label to be False"
    helper[current_index]['children'].append(len(helper))
    helper = build_tree_helper(tree_node_json['children'][1], helper)

    return helper

def parse_tree_helper_json(tree_helper_path):
    with open(tree_helper_path, 'r') as file:
        tree_helper = json.load(file)
    tree_helper =  build_tree_helper(tree_helper, [])
    return tree_helper


def run_dtcontrol(scheduler_represenation, representation_file_type, metadata=None, preset="default", show_stdout=False):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    temp_file_name = "subtree_test" + timestamp
    try:
        os.makedirs(temp_file_name, exist_ok=True)
        open(f"{temp_file_name}/scheduler.{representation_file_type}", "w").write(scheduler_represenation)

        if metadata is not None:
            open(f"{temp_file_name}/scheduler_config.json", "w").write(metadata)

        command = ["dtcontrol", "--input", f"scheduler.{representation_file_type}", "-r", "--use-preset", preset]
        if show_stdout:
            subprocess.run(command, cwd=f"{temp_file_name}")
        else:
            subprocess.run(command, cwd=f"{temp_file_name}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info(f"parsing new dtcontrol tree for setting {preset}")
        dtcontrol_tree_helper = parse_tree_helper_json(f"{temp_file_name}/decision_trees/{preset}/scheduler/{preset}.json")

        shutil.rmtree(f"{temp_file_name}")
    except:
        shutil.rmtree(f"{temp_file_name}")
        raise Exception("error when calling dtcontrol. Possible KeyboardInterrupt?")
    
    return dtcontrol_tree_helper


def state_to_choice_to_state_to_action(state_to_choice, quotient):
    state_to_action = []
    for state in range(quotient.quotient_mdp.nr_states):
        if state_to_choice[state] is None or not quotient.state_is_relevant_bv.get(state):
            state_to_action.append(-1)
        else:
            state_to_action.append(quotient.choice_to_action[state_to_choice[state]])

    return state_to_action

def scikit_tree_to_tree_helper(clf, variables, action_labels):
    helper = []
    num_nodes = clf.tree_.node_count

    for i in range(num_nodes):
        if clf.tree_.children_left[i] == -1: # leaf node
            chosen_idx = clf.tree_.value[i].argmax()
            # Assert that value is 1 for the chosen action and 0 for all others
            values = clf.tree_.value[i][0]
            assert (values == 1).sum() == 1 and (values == 0).sum() == (len(values) - 1), (
                f"Expected one value to be 1 and the rest 0, got {values}")
            helper.append({'id': i, 'leaf': True, 'chosen': [action_labels[chosen_idx]]})
            continue

        variable = variables[clf.tree_.feature[i]].name
        threshold = floor(clf.tree_.threshold[i])

        helper.append({'id': i, 'leaf': False, 'chosen': (variable, threshold), 'children': [int(clf.tree_.children_left[i]), int(clf.tree_.children_right[i])]})

    return helper

def run_scikit_learn_tree(state_valuations, state_to_action, variables, action_labels, filter_unreachable=True, max_depth=None, min_samples_leaf=1):

    X = state_valuations
    Y = state_to_action

    if filter_unreachable:
        X = [x for j, x in enumerate(state_valuations) if state_to_action[j] != -1]
        Y = [y for j, y in enumerate(state_to_action) if state_to_action[j] != -1]

    adjusted_action_labels = [x for i, x in enumerate(action_labels) if i in Y]

    clf = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    clf = clf.fit(X, Y)

    scikit_tree_helper = scikit_tree_to_tree_helper(clf, variables, adjusted_action_labels)
    
    return scikit_tree_helper