"""Telling apart the two ways a colored MDP's choices can be colored: by explicit (parameter, option) pairs, as in a payntbind Coloring, or by an arbitrary
formula, as in a ColoringGeneral (see paynt.utils.coloring_builder), which has no pairs to read back -- and reading such a formula back as a z3py formula over
the parameters, see choice_formula, parameter_variables and parameter_space_formula.

Where the pairs are needed, see
paynt.utils.error_handling.require_pair_list_coloring.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import z3

import payntbind.synthesis

if TYPE_CHECKING:
    import paynt.parameter_space.parameter_space

NodeOp = payntbind.synthesis.ColoringGeneralNodeOp

_ARITHMETIC = {NodeOp.AddTerm: operator.add, NodeOp.SubTerm: operator.sub, NodeOp.MulTerm: operator.mul}
_COMPARISON = {
    NodeOp.EqF: operator.eq,
    NodeOp.NeF: operator.ne,
    NodeOp.LtF: operator.lt,
    NodeOp.LeF: operator.le,
    NodeOp.GtF: operator.gt,
    NodeOp.GeF: operator.ge,
}
# the nodes of the node table with two operands that are nodes themselves (NotF and InSetF have one, IteTerm and IteF three)
_BINARY = {NodeOp.AndF, NodeOp.OrF, NodeOp.ImpliesF, NodeOp.InBitsF, *_ARITHMETIC, *_COMPARISON}


def is_general_coloring(coloring: Any) -> bool:
    """Whether coloring is a payntbind ColoringGeneral rather than a coloring given by explicit (parameter, option) pairs."""
    return isinstance(coloring, payntbind.synthesis.ColoringGeneral)


def parameter_variables(parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> list[z3.ArithRef]:
    """One z3 Int for every parameter of the parameter space, named as the parameter: the variables that choice_formula speaks about, whose value is the index
    of an option.

    A ParameterSpace allows two parameters to have the same name (the POMDP factories make them from observations), and Z3 identifies a variable by its name, so
    they would be one variable. If any two share a name, every variable is named f"{name}#{index}" instead, as in
    paynt.parameter_space.bitvec.parameter_bitvec_variables (whose variables are BitVecs, for the SMT encodings of the engines).
    """
    names = [parameter_space.parameter_name(parameter) for parameter in range(parameter_space.num_parameters)]
    if len(set(names)) < len(names):
        names = [f"{name}#{parameter}" for parameter, name in enumerate(names)]
    return [z3.Int(name) for name in names]


def parameter_space_formula(parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> z3.BoolRef:
    """The parameter space as a z3py formula over parameter_variables: every parameter takes one of its options.

    The options are those that the parameter space has now, so for a family (a parameter space restricted to some of the options of its parameters) it is the
    formula of the family. With the formulas of choices (choice_formula), it tells the assignments that enable all of them at once; for the choices of a
    scheduler, whether the coloring can make it, which ColoringGeneral.areChoicesConsistent decides in C++.

    Example, to ask for an assignment that enables some choices at once, and to read it back:

        space = colored_mdp.parameter_space
        solver = z3.Solver()
        solver.add(parameter_space_formula(space), *(choice_formula(colored_mdp.coloring, choice, space) for choice in choices))
        if solver.check() == z3.sat:
            model = solver.model()
            options = [model.eval(variable, model_completion=True).as_long() for variable in parameter_variables(space)]
            print(space.construct_assignment(options))
    """
    clauses: list[Any] = []
    for parameter, variable in enumerate(parameter_variables(parameter_space)):
        runs = _runs(parameter_space.parameter_options(parameter))
        if len(runs) == 1:
            first, last = runs[0]
            clauses.extend([variable == first] if first == last else [variable >= first, variable <= last])
        else:
            alternatives = [variable == first if first == last else z3.And(variable >= first, variable <= last) for first, last in runs]
            clauses.append(z3.Or(alternatives) if alternatives else z3.BoolVal(False))
    return z3.And(clauses) if len(clauses) > 1 else clauses[0] if clauses else z3.BoolVal(True)


def _runs(options: list[int]) -> list[tuple[int, int]]:
    """The maximal runs of consecutive numbers among the options, each as (first, last): a parameter with a thousand options is a range, not a disjunction."""
    runs: list[list[int]] = []
    for option in sorted(set(options)):
        if runs and option == runs[-1][1] + 1:
            runs[-1][1] = option
        else:
            runs.append([option, option])
    return [(first, last) for first, last in runs]


def choice_formula(coloring: Any, choice: int, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace | None = None) -> z3.BoolRef:
    """The color of a choice of a general coloring, as a z3py formula over the parameters, which are named as in the parameter space if it is given (see
    parameter_variables), and p0, p1, ... if not.

    :param coloring: a payntbind ColoringGeneral, e.g. the coloring of a ColoredMdp for which has_general_coloring holds
    :param choice: a choice of the underlying MDP
    :param parameter_space: the parameter space that the coloring is defined over, e.g. that of the ColoredMdp, which has the names of the parameters; without
        it, the variables are z3.Int("p0"), z3.Int("p1"), ... for the parameters 0, 1, ...
    :raises TypeError: if the coloring is not a general one
    :raises IndexError: if the coloring has no such choice
    :raises ValueError: if the color depends on more parameters than the parameter space has
    """
    if not is_general_coloring(coloring):
        raise TypeError(
            f"the formula of a choice is read from a general coloring (payntbind.synthesis.ColoringGeneral), not from a {type(coloring).__name__}: the color "
            "of a choice of a standard coloring is the conjunction of its (parameter, option) pairs, see Coloring.getChoiceToAssignment"
        )
    root, state_row, choice_row, state_is_relevant, is_first_of_state = coloring.getChoiceColor(choice)
    if not state_is_relevant:
        return z3.BoolVal(is_first_of_state)
    if root < 0:
        return z3.BoolVal(True)

    variables = None if parameter_space is None else parameter_variables(parameter_space)

    def variable(index: int) -> Any:
        if variables is None:
            return z3.Int(f"p{index}")
        if index >= len(variables):
            raise ValueError(
                f"the color of choice {choice} depends on parameter {index}, but the parameter space has only {len(variables)} parameters: is it the one that "
                "the coloring is defined over?"
            )
        return variables[index]

    formula = _rebuild_formula(coloring.getNodeTable(), root, state_row, choice_row, variable)
    return z3.BoolVal(formula) if isinstance(formula, bool) else z3.simplify(formula)


def _operands(op: Any, a: int, b: int, c: int) -> tuple[int, ...]:
    """The operands (a, b, c) of a node of the node table that are nodes themselves: the others are a number, a parameter or an option set."""
    if op in (NodeOp.IteTerm, NodeOp.IteF):
        return (a, b, c)
    if op in _BINARY:
        return (a, b)
    if op in (NodeOp.NotF, NodeOp.InSetF):
        return (a,)
    return ()


def _rebuild_formula(table: tuple[Any, ...], root: int, state_row: list[int], choice_row: list[int], variable: Any) -> Any:
    """The formula of the node root of the node table of a general coloring, with the data of a state and of a choice substituted into it: a z3 formula, or a
    Python bool if it depends on no parameter. On the way, the terms that depend on none are Python ints and bools, so that no z3 term is made of them.

    The node table is a DAG that is built bottom-up, so a node has a larger id than its operands, and the formula is rebuilt from the leaves up without any
    recursion. A recursive walk would exceed Python's limit on the chain of disjuncts that a decision tree of depth 10 makes, one for each of its leaves.

    Every ColoringGeneral::NodeOp needs a case here.
    """
    ops, node_a, node_b, node_c, option_sets = table
    needed = {root}
    order = []
    for node in range(root, -1, -1):
        if node in needed:
            order.append(node)
            needed.update(_operands(NodeOp(ops[node]), node_a[node], node_b[node], node_c[node]))

    values: dict[int, Any] = {}
    for node in reversed(order):
        op, a, b, c = NodeOp(ops[node]), node_a[node], node_b[node], node_c[node]
        if op == NodeOp.ConstTerm:
            value = a
        elif op == NodeOp.ParamTerm:
            value = variable(a)
        elif op == NodeOp.StateColTerm:
            value = state_row[a]
        elif op == NodeOp.ChoiceColTerm:
            value = choice_row[a]
        elif op in _ARITHMETIC:
            value = _ARITHMETIC[op](values[a], values[b])
        elif op in (NodeOp.IteTerm, NodeOp.IteF):
            value = _if(values[a], values[b], values[c])
        elif op == NodeOp.TrueF:
            value = True
        elif op == NodeOp.FalseF:
            value = False
        elif op == NodeOp.NotF:
            value = _not(values[a])
        elif op == NodeOp.AndF:
            value = _and(values[a], values[b])
        elif op == NodeOp.OrF:
            value = _or(values[a], values[b])
        elif op == NodeOp.ImpliesF:
            value = _or(_not(values[a]), values[b])
        elif op in _COMPARISON:
            value = _COMPARISON[op](values[a], values[b])
        elif op == NodeOp.InSetF:
            value = _any_of(values[a], option_sets[b])
        else:  # InBitsF: the parameter is one of the set bits of a data word, which depends on no parameter (a negative word has its bit 63 set)
            if not isinstance(values[b], int):
                raise ValueError("the data operand of an in_bits atom depends on a parameter")
            value = _any_of(values[a], [option for option in range(64) if (values[b] >> option) & 1])
        values[node] = value
    return values[root]


# The connectives below take z3 formulas and Python bools, and answer with a bool wherever the value is known.


def _not(formula: Any) -> Any:
    return (not formula) if isinstance(formula, bool) else z3.Not(formula)


def _and(left: Any, right: Any) -> Any:
    if left is False or right is False:
        return False
    if left is True:
        return right
    if right is True:
        return left
    return z3.And(left, right)


def _or(left: Any, right: Any) -> Any:
    if left is True or right is True:
        return True
    if left is False:
        return right
    if right is False:
        return left
    return z3.Or(left, right)


def _if(condition: Any, then: Any, otherwise: Any) -> Any:
    """If condition then then else otherwise, for terms and for formulas."""
    if condition is True:
        return then
    if condition is False:
        return otherwise
    return z3.If(condition, then, otherwise)


def _any_of(variable: Any, options: list[int]) -> Any:
    """The formula that the variable is one of the options."""
    return z3.Or([variable == option for option in options]) if options else False


class ChoiceCoverage:
    """Which states every design of a parameter space keeps a choice in.

    A sub-MDP of several designs holds every choice that one of them takes, so a state where some design keeps none still has the choices of the others: the
    sub-MDP says nothing about the design that gets stuck there. The states that some design of the whole parameter space leaves without a choice are found
    once; only those are looked at again for a parameter space.
    """

    def __init__(self, colored_mdp: Any):
        self.coloring = colored_mdp.coloring
        self.row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
        self.general = is_general_coloring(self.coloring)
        self.incomplete = self._without_a_choice(colored_mdp.parameter_space, range(colored_mdp.underlying_mdp.nr_states))

    def states_without_a_choice(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, states: Any) -> list[int]:
        """The given states in which some design of parameter_space keeps no choice."""
        if not self.incomplete:
            return []
        states = set(states)
        return self._without_a_choice(parameter_space, [state for state in self.incomplete if state in states])

    def _without_a_choice(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, states: Any) -> list[int]:
        if not self.general:
            return list(self.coloring.statesWithoutChoice(parameter_space.native, list(states)))
        domain = parameter_space_formula(parameter_space)
        without = []
        for state in states:
            colors = [choice_formula(self.coloring, choice, parameter_space) for choice in range(self.row_groups[state], self.row_groups[state + 1])]
            solver = z3.Solver()
            solver.add(domain, z3.Not(z3.Or(colors)))
            if solver.check() == z3.sat:
                without.append(state)
        return without
