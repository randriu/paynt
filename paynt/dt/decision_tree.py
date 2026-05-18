from collections.abc import Callable
from typing import Optional

import graphviz

import logging
logger = logging.getLogger(__name__)

class DtVariable:

    def __init__(self, name : str, domain : set[int]):
        self.name = name
        # conversion of boolean variables to integers
        domain_new = []
        for value in domain:
            if value is True:
                value = 1
            elif value is False:
                value = 0
            domain_new.append(value)
        domain = domain_new
        domain = sorted(domain)
        self.domain = domain

    @property
    def domain_min(self) -> int:
        return self.domain[0]

    @property
    def domain_max(self) -> int:
        return self.domain[-1]

    @property
    def parameter_domain(self) -> list[int]:
        '''
        Parameter domain does not include the maximum value.
        '''
        return self.domain[:-1]

    def __str__(self) -> str:
        domain = f"[{self.domain_min}..{self.domain_max}]"
        return f"{self.name}:{domain}"




class DecisionTreeNode:

    def __init__(self, parent : Optional["DecisionTreeNode"]):
        self.parent = parent
        self.child_true = None
        self.child_false = None
        self.identifier = None
        self.parameters = None

        self.action = None
        self.variable = None
        self.variable_bound = None

    @property
    def is_terminal(self) -> bool:
        return self.child_false is None and self.child_true is None

    @property
    def child_nodes(self) -> list["DecisionTreeNode"]:
        return [] if self.is_terminal else [self.child_true,self.child_false]

    @property
    def is_true_child(self) -> bool:
        return self is self.parent.child_true

    def add_children(self):
        assert self.is_terminal
        self.child_true = DecisionTreeNode(self)
        self.child_false = DecisionTreeNode(self)

    def get_depth(self) -> int:
        if self.is_terminal:
            return 0
        return 1 + max([child.get_depth() for child in self.child_nodes])
    
    def set_node_from_helper(self, tree_helper_node, variable_names, variable_domains, action_labels):
        if tree_helper_node["leaf"]:
            self.action = action_labels.index(tree_helper_node["chosen"][0])
        else:
            self.variable = variable_names.index(tree_helper_node["chosen"][0])
            # dtControl uses values that are not necessarily in the domain, e.g. let's say our domain is [0,2,3]
            # and we want to split [0] and [2,3], dtControl can choose 0.5 or 1.5. We expect it will be 0 in this case.
            # Note that dtControl bound decisions are floored when creating the tree helper i.e. 1.5 becomes 1
            while tree_helper_node["chosen"][1] not in variable_domains[self.variable]:
                tree_helper_node["chosen"] = (tree_helper_node["chosen"][0], tree_helper_node["chosen"][1]-1)
            self.variable_bound = variable_domains[self.variable].index(tree_helper_node["chosen"][1])

    def add_children_from_helper(self, tree_helper, helper_node, variable_names, variable_domains, action_labels):
        assert self.is_terminal
        self.child_true = DecisionTreeNode(self)
        true_child_helper = tree_helper[helper_node["children"][0]]
        self.child_true.set_node_from_helper(true_child_helper, variable_names, variable_domains, action_labels)
        self.child_false = DecisionTreeNode(self)
        false_child_helper = tree_helper[helper_node["children"][1]]
        self.child_false.set_node_from_helper(false_child_helper, variable_names, variable_domains, action_labels)

    def get_number_of_descendants(self) -> int:
        if self.is_terminal:
            return 0
        return 1 + sum([child.get_number_of_descendants() for child in self.child_nodes])

    def assign_identifiers(self, identifier : int = 0, keep_old : bool = False) -> int:
        '''
        Recursively assigns unique identifiers to each child node with pre-order traversal.
            If keep_old is True, the old identifier is stored in the old_identifier attribute before being overwritten.
        '''
        if keep_old:
            self.old_identifier = self.identifier
        self.identifier = identifier
        if self.is_terminal:
            return self.identifier
        identifier = self.child_true.assign_identifiers(identifier+1, keep_old=keep_old)
        identifier = self.child_false.assign_identifiers(identifier+1, keep_old=keep_old)
        return identifier

    def associate_parameters(self, node_parameter_info : list[tuple[int,str,str]]):
        self.parameters = [parameter for parameter,_,_ in node_parameter_info[self.identifier]]
        if self.is_terminal:
            return
        self.child_true.associate_parameters(node_parameter_info)
        self.child_false.associate_parameters(node_parameter_info)

    def associate_assignment(self, assignment): # TODO type of assignment
        parameter_assignment = [assignment.hole_options(parameter)[0] for parameter in self.parameters]
        if self.is_terminal:
            self.action = parameter_assignment[0]
            return

        self.variable = parameter_assignment[0]
        self.variable_bound = parameter_assignment[self.variable+1]

        self.child_true.associate_assignment(assignment)
        self.child_false.associate_assignment(assignment)

    def apply_hint(self, subfamily, tree_hint):
        if self.is_terminal or tree_hint.is_terminal:
            return

        variable_hint = tree_hint.variable
        subfamily.hole_set_options(self.parameters[0],[variable_hint]) # TODO refactor, rename holes to parameters
        subfamily.hole_set_options(self.parameters[variable_hint+1],[tree_hint.variable_bound])
        self.child_true.apply_hint(subfamily,tree_hint.child_true)
        self.child_false.apply_hint(subfamily,tree_hint.child_false)

    def simplify(self, variables : list["DtVariable"], state_valuations : list[list[int]]):
        '''
        Recursively simplifies the decision tree by removing nodes that have only one child due to the given state valuations, and by merging terminal nodes with the same action.
        '''
        if self.is_terminal:
            return

        bound = variables[self.variable].domain[self.variable_bound]
        state_valuations_true =  [valuation for valuation in state_valuations if valuation[self.variable] <= bound]
        state_valuations_false = [valuation for valuation in state_valuations if valuation[self.variable]  > bound]
        child_skip = None
        if len(state_valuations_true) == 0:
            child_skip = self.child_false
        elif len(state_valuations_false) == 0:
            child_skip = self.child_true
        if child_skip is not None:
            self.variable = child_skip.variable
            self.variable_bound = child_skip.variable_bound
            self.action = child_skip.action
            self.child_true = child_skip.child_true
            self.child_false = child_skip.child_false
            self.simplify(variables,state_valuations)
            return

        self.child_true.simplify(variables, state_valuations_true)
        self.child_false.simplify(variables, state_valuations_false)
        if not self.is_terminal and self.child_true.is_terminal and self.child_false.is_terminal and self.child_true.action == self.child_false.action:
            self.variable = self.variable_bound = None
            self.action = self.child_true.action
            self.child_true = self.child_false = None

    def branch_expression(self, variables : list["DtVariable"], true_branch : bool = True) -> str:
        var = variables[self.variable]
        if true_branch:
            return f"{var.name}<={var.domain[self.variable_bound]}"
        else:
            return f"{var.name}>{var.domain[self.variable_bound]}"

    def path_expression(self, variables : list["DtVariable"]) -> list[str]:
        if self.parent is None:
            return []
        return self.parent.path_expression(variables) + [self.parent.branch_expression(variables,true_branch=self.is_true_child)]
    
    def copy(self, parent):
        node_copy = DecisionTreeNode(parent)
        node_copy.identifier = self.identifier
        node_copy.parameters = self.parameters
        if self.is_terminal:
            node_copy.action = self.action
        else:
            node_copy.variable = self.variable
            node_copy.variable_bound = self.variable_bound
            node_copy.child_true = self.child_true.copy(node_copy)
            node_copy.child_false = self.child_false.copy(node_copy)
        return node_copy

    def to_string(self, variables : list["DtVariable"], action_labels : list[str], indent_level : int = 0, indent_size : int = 2) -> str:
        indent = " "*indent_level*indent_size
        if self.is_terminal:
            return indent + f"{action_labels[self.action]}" + "\n"
        s = ""
        s += indent + f"if {self.branch_expression(variables)}:" + "\n"
        s += self.child_true.to_string(variables,action_labels,indent_level+1)
        s += indent + f"else:" + "\n"
        s += self.child_false.to_string(variables,action_labels,indent_level+1)
        return s

    @property
    def graphviz_id(self) -> str:
        return str(self.identifier)

    def to_graphviz(self, graphviz_tree : graphviz.Digraph, variables : list["DtVariable"], action_labels : list[str], highlight_nodes : list[int] = []):
        if not self.is_terminal:
            for child in self.child_nodes:
                child.to_graphviz(graphviz_tree,variables,action_labels,highlight_nodes)

        if self.is_terminal:
            node_label = action_labels[self.action]
        else:
            var = variables[self.variable]
            node_label = f"{var.name}<={var.domain[self.variable_bound]}"

        if self.identifier in highlight_nodes:
            graphviz_tree.node(self.graphviz_id, label=node_label, shape="box", style="filled", fillcolor="lightgreen", margin="0.05,0.05")
        else:
            graphviz_tree.node(self.graphviz_id, label=node_label, shape="box", style="rounded", margin="0.05,0.05")
        if not self.is_terminal:
            graphviz_tree.edge(self.graphviz_id,self.child_true.graphviz_id,label="T")
            graphviz_tree.edge(self.graphviz_id,self.child_false.graphviz_id,label="F")
        
    
    # in dtNest the tree nodes contain reference indeces to objects from subtree_quotient
    # this needs to be fixed to match the references in the original quotient
    def fix_with_respect_to_quotient(self, action_labels, new_action_labels, variables, new_variables):
        if self.is_terminal:
            old_index = self.action
            self.action = new_action_labels.index(action_labels[old_index])
            return
        
        var = variables[self.variable]
        var_name = var.name
        var_bound = var.domain[self.variable_bound]
        for var_id, new_var in enumerate(new_variables):
            if new_var.name == var_name:
                break
        bound_id = new_var.domain.index(var_bound)
        self.variable = var_id
        self.variable_bound = bound_id

        self.child_true.fix_with_respect_to_quotient(action_labels, new_action_labels, variables, new_variables)
        self.child_false.fix_with_respect_to_quotient(action_labels, new_action_labels, variables, new_variables)



class DecisionTree:

    def __init__(self, action_labels : list[str], variables : list["DtVariable"]):
        self.action_labels = action_labels
        self.variables = variables
        self.reset()

    def reset(self):
        self.root = DecisionTreeNode(None)

    def random_tree(self):
        self.root = DecisionTreeNode(None)
        self.root.action = self.action_labels.index("__random__")

    def copy(self):
        new_tree = DecisionTree(self.action_labels, self.variables)
        new_tree.root = self.root.copy(None)
        return new_tree

    def set_depth(self, depth : int):
        self.reset()
        for level in range(depth):
            for node in self.collect_terminals():
                node.add_children()
        self.root.assign_identifiers()

    def get_depth(self) -> int:
        return self.root.get_depth()
    
    def build_from_tree_helper(self, tree_helper):
        self.reset()
        node_stack = [(0, self.root)]
        var_names = [v.name for v in self.variables]
        var_domains = [v.domain for v in self.variables]
        self.root.set_node_from_helper(tree_helper[0], var_names, var_domains, self.action_labels)
        while node_stack:
            node_id, node = node_stack.pop()
            if tree_helper[node_id]["leaf"]:
                continue
            node.add_children_from_helper(tree_helper, tree_helper[node_id], var_names, var_domains, self.action_labels)
            node_stack.append((tree_helper[node_id]["children"][0], node.child_true))
            node_stack.append((tree_helper[node_id]["children"][1], node.child_false))

        self.root.assign_identifiers()

        # double check if identifiers correspond to the tree helper
        nodes = self.collect_nodes()
        for node_id, node in enumerate(nodes):
            assert node.is_terminal == tree_helper[node.identifier]["leaf"], f"node {node_id} is terminal: {node.is_terminal}, expected: {tree_helper[node_id]['leaf']}"
            assert node.identifier == tree_helper[node.identifier]["id"], f"node {node_id} has identifier {node.identifier}, expected: {tree_helper[node_id]['id']}"

    def collect_nodes(self, node_condition : Callable[[DecisionTreeNode], bool] | None = None) -> list[DecisionTreeNode]:
        '''
        Returns a list of nodes in the tree that satisfy the given node_condition. If node_condition is None, returns all nodes in the tree.
        '''
        if node_condition is None:
            node_condition = lambda node : True
        node_queue = [self.root]
        output_nodes = []
        while node_queue:
            node = node_queue.pop(0)
            if node_condition(node):
                output_nodes.append(node)
            node_queue += node.child_nodes
        return output_nodes

    def collect_terminals(self) -> list[DecisionTreeNode]:
        return self.collect_nodes(lambda node : node.is_terminal)

    def collect_nonterminals(self) -> list[DecisionTreeNode]:
        return self.collect_nodes(lambda node : not node.is_terminal)

    def to_list(self) -> list[tuple[int,int,int]]:
        num_nodes = len(self.collect_nodes())
        node_info = [ None for node in range(num_nodes) ]
        for node in self.collect_nodes():
            parent = num_nodes if node.parent is None else node.parent.identifier
            child_true = num_nodes if node.is_terminal else node.child_true.identifier
            child_false = num_nodes if node.is_terminal else node.child_false.identifier
            node_info[node.identifier] = (parent,child_true,child_false)
        return node_info

    def simplify(self, state_valuations : list[list[int]]):
        self.root.simplify(self.variables, state_valuations)

    def to_string(self) -> str:
        return self.root.to_string(self.variables,self.action_labels)

    def to_prism(self, indent_size : int = 2):
        indent = " "*indent_size
        s = ""
        s += "module scheduler\n"
        for terminal in self.collect_terminals():
            action = f"{self.action_labels[terminal.action]}"
            guard = " & ".join(terminal.path_expression(self.variables))
            if guard == "":
                guard = "true"
            s += f"{indent}[{action}] {guard} -> true;\n"
        # s += indent + f"if {var.name}<={var.domain[self.variable_bound]}:" + "\n"
        # s += self.child_true.to_string(variables,action_labels,indent_level+1)
        # s += indent + f"else:" + "\n"
        # s += self.child_false.to_string(variables,action_labels,indent_level+1)
        s += "endmodule\n"
        return s

    def to_graphviz(self, highlight_nodes : list[int] = []) -> graphviz.Digraph:
        logging.getLogger("graphviz").setLevel(logging.WARNING)
        logging.getLogger("graphviz.sources").setLevel(logging.ERROR)
        graphviz_tree = graphviz.Digraph(comment="decision tree")
        self.root.to_graphviz(graphviz_tree,self.variables,self.action_labels, highlight_nodes)
        return graphviz_tree
    
    def append_tree_as_subtree(self, new_subtree, subtree_root_node_id, subtree_quotient):
        subtree_root_node = self.collect_nodes(lambda node : node.identifier == subtree_root_node_id)
        assert len(subtree_root_node) == 1, f"subtree root node id {subtree_root_node_id} not found in decision tree"
        subtree_root_node = subtree_root_node[0]

        all_current_nodes = self.collect_nodes()
        new_subtree.root.assign_identifiers(identifier=len(all_current_nodes)+1)
        new_subtree.root.fix_with_respect_to_quotient(subtree_quotient.action_labels, self.action_labels, subtree_quotient.variables, self.variables)

        parent = subtree_root_node.parent
        if parent.child_true.identifier == subtree_root_node.identifier:
            parent.child_true = new_subtree.root
        else:
            parent.child_false = new_subtree.root