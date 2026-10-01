"""Builder for colorings given by formulas: the authoring side of payntbind.synthesis.Coloring and payntbind.synthesis.ColoringGeneral.

The standard Coloring gives every choice a conjunction of (parameter = option) pairs. A ColoringGeneral (SMPMC paper,
arXiv:2511.08078, Definition 2) instead colors every choice with an arbitrary quantifier-free formula over the finite
parameter space: the choice is enabled under a full parameter assignment iff the assignment satisfies its formula.
ColoringBuilder is how either is defined, entirely in Python -- the C++ engine is generic and never needs touching to
support a new kind of coloring. A color is written once, as a formula, and build() produces the standard coloring whenever
every color is a conjunction of (parameter = option) pairs -- the cheap representation every engine understands -- and the
general one otherwise.

Colors are written as ordinary z3py formulas over the coloring's own parameter variables (one per declared parameter,
value = option index) and optional per-state/per-choice data columns, and ColoringBuilder translates them into the flat
node table ColoringGeneral's constructor expects. No z3 term ever crosses into C++ (payntbind links a different Z3
build than z3py, so the two cannot share objects): the walker below reads a z3py AST and re-emits it as plain data.

Supported formula language (anything else raises ValueError with the offending sub-expression):
    formulas: True, False, Not, And, Or (any arity), Implies, If (boolean result), ==, !=, <, <=, >, >=
    terms:    integer literals, parameter variables, data columns, +, -, unary -, *, If (integer result)
    atoms:    Or([p == o1, p == o2, ...]) for a single bare parameter p is recognised and compiled to an exact
              "p in {o1,o2,...}" atom instead of a chain of (possibly imprecise, see ColoringGeneral) equalities;
              ColoringBuilder.in_bits(p, data) similarly compiles to an exact "p's value is a set bit of the
              (per-state/choice) data word" atom (data is interpreted as a 64-bit bitmask, one bit per option).

Example: "the environment parameter is 0 or 2, or the state's flag is set" for choices [3, 7], and "the shared-action parameter is 'left'" for choice 5,
over an existing parameter_space:

    builder = ColoringBuilder(colored_mdp.underlying_mdp.nondeterministic_choice_indices, parameter_space.num_parameters)
    p, action = builder.parameters[0], builder.parameters[1]
    flag = builder.state_column([...])  # one entry per state
    builder.color([3, 7], builder.template(z3.Or(z3.Or(p == 0, p == 2), flag == 1)))
    builder.color(5, builder.template(action == 2))
    # every other choice is left uncolored (always enabled), matching Coloring's own convention
    coloring = builder.build(parameter_space)  # ColoringGeneral: the first color is not a conjunction of (parameter = option) pairs
    colored_mdp = paynt.colored_mdp.ColoredMdp(colored_mdp.underlying_mdp, parameter_space, coloring)

Had every color been such a conjunction -- `action == 2`, `z3.And(p == 0, action == 2)`, or `action == choice_action` for a per-choice data column -- build()
would have returned the standard Coloring instead, which also works with generic AR and Hybrid. A colored MDP built from a ColoringGeneral works with the
OneByOne, CEGIS and SMPMC engines; generic AR and Hybrid read (parameter, option) pairs back from the coloring and are rejected (see
ColoredMdp.has_general_coloring).

A coloring need not enable exactly one choice in every state: leaving choices uncolored (always enabled) next to others gives an incomplete coloring, whose
full assignments are MDPs that engines check for the best resolution of the choices left -- say, "the relevant states with x == 5 play the same action", which
is `builder.share_action(parameter, states, choice_to_action)`. That is supported for a single property, and every reachable state must keep at least one choice
(check_definition(..., complete=False) tests both the parameter space and the colors). See paynt.colored_mdp.

The builder itself only needs a parameter *count* (a coloring built from a structural template, e.g. a decision tree, determines its own parameter layout from
the template, so its ParameterSpace can only be reconstructed by the caller afterwards -- see paynt/dt/coloring_general.py and DtColoredMdpFactory.reset_tree),
and such a coloring is always general. The ParameterSpace is given to build() only to get the standard coloring, which needs the number of options of every
parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import z3

import payntbind.synthesis
import paynt.parameter_space.parameter_space

import logging

logger = logging.getLogger(__name__)

NodeOp = payntbind.synthesis.ColoringGeneralNodeOp

# a marker function used purely as syntax: never handed to a solver, only ever pattern-matched by the walker below
_IN_BITS_FUNC = z3.Function("__paynt_coloring_builder_in_bits__", z3.IntSort(), z3.IntSort(), z3.BoolSort())

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _check_fits_int64(values: list[int], kind: str) -> None:
    """Data columns reach the engine as signed 64-bit integers: refuse a wider value here, rather than letting pybind fail on it with an opaque TypeError."""
    if values and (min(values) < _INT64_MIN or max(values) > _INT64_MAX):
        index = next(index for index, value in enumerate(values) if not _INT64_MIN <= value <= _INT64_MAX)
        raise ValueError(
            f"{kind} value {values[index]} at index {index} does not fit in a signed 64-bit integer, which is what a data column holds "
            "(so an in_bits mask can address options 0..63, bit 63 being the sign bit)"
        )


class _NotStandard(Exception):
    """A color that the standard coloring cannot represent; the message says why."""


class _NodeTable:
    """Hash-consed flat node table (ColoringGeneral's constructor format), shared by every template of one builder."""

    def __init__(self) -> None:
        self.op: list[Any] = []
        self.a: list[int] = []
        self.b: list[int] = []
        self.c: list[int] = []
        self.option_sets: list[list[int]] = []
        self._node_cache: dict[tuple[Any, int, int, int], int] = {}
        self._option_set_cache: dict[tuple[int, ...], int] = {}

    def _add(self, op: Any, a: int, b: int, c: int) -> int:
        key = (op, a, b, c)
        node = self._node_cache.get(key)
        if node is not None:
            return node
        node = len(self.op)
        self.op.append(op)
        self.a.append(a)
        self.b.append(b)
        self.c.append(c)
        self._node_cache[key] = node
        return node

    def option_set(self, options: Iterable[int]) -> int:
        key = tuple(sorted({int(option) for option in options}))
        index = self._option_set_cache.get(key)
        if index is not None:
            return index
        index = len(self.option_sets)
        self.option_sets.append(list(key))
        self._option_set_cache[key] = index
        return index

    def const(self, value: int) -> int:
        return self._add(NodeOp.ConstTerm, int(value), 0, 0)

    def param(self, index: int) -> int:
        return self._add(NodeOp.ParamTerm, int(index), 0, 0)

    def state_col(self, column: int) -> int:
        return self._add(NodeOp.StateColTerm, int(column), 0, 0)

    def choice_col(self, column: int) -> int:
        return self._add(NodeOp.ChoiceColTerm, int(column), 0, 0)

    def add(self, a: int, b: int) -> int:
        return self._add(NodeOp.AddTerm, a, b, 0)

    def sub(self, a: int, b: int) -> int:
        return self._add(NodeOp.SubTerm, a, b, 0)

    def mul(self, a: int, b: int) -> int:
        return self._add(NodeOp.MulTerm, a, b, 0)

    def ite_term(self, cond: int, then: int, els: int) -> int:
        return self._add(NodeOp.IteTerm, cond, then, els)

    def true(self) -> int:
        return self._add(NodeOp.TrueF, 0, 0, 0)

    def false(self) -> int:
        return self._add(NodeOp.FalseF, 0, 0, 0)

    def not_(self, a: int) -> int:
        return self._add(NodeOp.NotF, a, 0, 0)

    def and_(self, a: int, b: int) -> int:
        return self._add(NodeOp.AndF, a, b, 0)

    def or_(self, a: int, b: int) -> int:
        return self._add(NodeOp.OrF, a, b, 0)

    def implies(self, a: int, b: int) -> int:
        return self._add(NodeOp.ImpliesF, a, b, 0)

    def ite_formula(self, cond: int, then: int, els: int) -> int:
        return self._add(NodeOp.IteF, cond, then, els)

    def eq(self, a: int, b: int) -> int:
        return self._add(NodeOp.EqF, a, b, 0)

    def ne(self, a: int, b: int) -> int:
        return self._add(NodeOp.NeF, a, b, 0)

    def lt(self, a: int, b: int) -> int:
        return self._add(NodeOp.LtF, a, b, 0)

    def le(self, a: int, b: int) -> int:
        return self._add(NodeOp.LeF, a, b, 0)

    def gt(self, a: int, b: int) -> int:
        return self._add(NodeOp.GtF, a, b, 0)

    def ge(self, a: int, b: int) -> int:
        return self._add(NodeOp.GeF, a, b, 0)

    def in_set(self, param_node: int, options: Iterable[int]) -> int:
        return self._add(NodeOp.InSetF, param_node, self.option_set(options), 0)

    def in_bits(self, param_node: int, data_node: int) -> int:
        return self._add(NodeOp.InBitsF, param_node, data_node, 0)


class Template:
    """A compiled color, ready to be attached to one or more choices via ColoringBuilder.color()."""

    def __init__(self, node: int) -> None:
        self.node = node


class _Walker:
    """Translates one z3py formula into ColoringBuilder's shared node table."""

    def __init__(self, builder: ColoringBuilder) -> None:
        self.builder = builder
        self.table = builder._table

    def _leaf(self, e: Any) -> int | None:
        name = str(e)
        if name in self.builder._param_index_by_name:
            return self.table.param(self.builder._param_index_by_name[name])
        if name in self.builder._state_col_by_name:
            return self.table.state_col(self.builder._state_col_by_name[name])
        if name in self.builder._choice_col_by_name:
            return self.table.choice_col(self.builder._choice_col_by_name[name])
        return None

    def _fold(self, children: list[Any], op: Any, walk: Any) -> int:
        node = walk(children[0])
        for child in children[1:]:
            node = op(node, walk(child))
        return node

    def _in_set_atom(self, e: Any) -> tuple[int, int] | None:
        """If e is (bare parameter) == (integer literal), in either order, return (its ParamTerm node id, the literal); else None -- used to recognise Or([p ==
        o, ...]) as an exact "p in {...}" atom."""
        if not z3.is_app(e) or e.decl().kind() != z3.Z3_OP_EQ:
            return None
        x, y = e.children()
        for lhs, rhs in ((x, y), (y, x)):
            leaf = self._leaf(lhs)
            if leaf is not None and self.table.op[leaf] == NodeOp.ParamTerm and z3.is_int_value(rhs):
                return leaf, rhs.as_long()
        return None

    def walk_term(self, e: Any) -> int:
        if z3.is_int_value(e):
            return self.table.const(e.as_long())
        leaf = self._leaf(e)
        if leaf is not None:
            return leaf
        if z3.is_app(e):
            kind = e.decl().kind()
            children = e.children()
            if kind == z3.Z3_OP_ADD:
                return self._fold(children, self.table.add, self.walk_term)
            if kind == z3.Z3_OP_SUB:
                if len(children) == 1:
                    return self.table.sub(self.table.const(0), self.walk_term(children[0]))
                return self._fold(children, self.table.sub, self.walk_term)
            if kind == z3.Z3_OP_UMINUS:
                return self.table.sub(self.table.const(0), self.walk_term(children[0]))
            if kind == z3.Z3_OP_MUL:
                return self._fold(children, self.table.mul, self.walk_term)
            if kind == z3.Z3_OP_ITE:
                cond, then, els = children
                return self.table.ite_term(self.walk_formula(cond), self.walk_term(then), self.walk_term(els))
        raise ValueError(f"unsupported term in a general coloring template: {e}")

    def walk_formula(self, e: Any) -> int:
        if z3.is_true(e):
            return self.table.true()
        if z3.is_false(e):
            return self.table.false()
        if not z3.is_app(e):
            raise ValueError(f"unsupported formula in a general coloring template: {e}")
        if e.decl() == _IN_BITS_FUNC:
            p_arg, d_arg = e.children()
            p_node = self.walk_term(p_arg)
            if self.table.op[p_node] != NodeOp.ParamTerm:
                raise ValueError(f"in_bits' first argument must be a bare parameter reference, got: {p_arg}")
            return self.table.in_bits(p_node, self.walk_term(d_arg))
        kind = e.decl().kind()
        children = e.children()
        if kind == z3.Z3_OP_NOT:
            return self.table.not_(self.walk_formula(children[0]))
        if kind == z3.Z3_OP_AND:
            return self._fold(children, self.table.and_, self.walk_formula)
        if kind == z3.Z3_OP_OR:
            in_set = self._recognise_in_set(children)
            if in_set is not None:
                return in_set
            return self._fold(children, self.table.or_, self.walk_formula)
        if kind == z3.Z3_OP_IMPLIES:
            a, b = children
            return self.table.implies(self.walk_formula(a), self.walk_formula(b))
        if kind == z3.Z3_OP_ITE:
            cond, then, els = children
            return self.table.ite_formula(self.walk_formula(cond), self.walk_formula(then), self.walk_formula(els))
        if kind == z3.Z3_OP_EQ:
            a, b = children
            return self.table.eq(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_DISTINCT:
            if len(children) != 2:
                raise ValueError("Distinct is only supported with exactly 2 arguments (use != instead)")
            a, b = children
            return self.table.ne(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_LT:
            a, b = children
            return self.table.lt(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_LE:
            a, b = children
            return self.table.le(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_GT:
            a, b = children
            return self.table.gt(self.walk_term(a), self.walk_term(b))
        if kind == z3.Z3_OP_GE:
            a, b = children
            return self.table.ge(self.walk_term(a), self.walk_term(b))
        raise ValueError(f"unsupported formula in a general coloring template: {e}")

    def _recognise_in_set(self, children: list[Any]) -> int | None:
        param_node: int | None = None
        options: list[int] = []
        for child in children:
            atom = self._in_set_atom(child)
            if atom is None:
                return None
            node, value = atom
            if param_node is None:
                param_node = node
            elif node != param_node:
                return None
            options.append(value)
        assert param_node is not None
        return self.table.in_set(param_node, options)


class ColoringBuilder:
    """Constructs a payntbind.synthesis.ColoringGeneral from z3py formulas -- see the module docstring."""

    def __init__(
        self,
        row_groups: list[int],
        num_parameters: int,
        state_is_relevant: Any = None,
    ) -> None:
        """
        :param row_groups: nondeterministic choice indices of the underlying quotient MDP (as on ColoredMdp.underlying_mdp)
        :param num_parameters: number of parameters colors are formulas over -- deliberately just a count, not a
            full ParameterSpace: a coloring built from a template (e.g. a decision tree, see paynt/dt/coloring_general.py)
            determines its own parameter *layout* (names, option counts) only from the coloring it just built, so a
            ParameterSpace is reconstructed by the caller only afterwards, never available to the builder itself
        :param state_is_relevant: optional stormpy/payntbind BitVector -- irrelevant states are never evaluated
            (see ColoringGeneral); every state is relevant by default
        """
        self.row_groups = list(row_groups)
        self.num_states = len(self.row_groups) - 1
        self.num_choices = self.row_groups[-1]
        self.num_parameters = num_parameters
        self._state_is_relevant = state_is_relevant
        self._table = _NodeTable()
        self._param_index_by_name: dict[str, int] = {}
        self._state_col_by_name: dict[str, int] = {}
        self._choice_col_by_name: dict[str, int] = {}
        self._state_rows: list[list[int]] = [[] for _ in range(self.num_states)]
        self._choice_rows: list[list[int]] = [[] for _ in range(self.num_choices)]
        self._choice_root: list[int | None] = [None] * self.num_choices
        self._parameters: list[Any] | None = None

    @property
    def parameters(self) -> list[Any]:
        """One z3py Int per declared parameter, value = option index."""
        if self._parameters is None:
            variables = []
            for index in range(self.num_parameters):
                name = f"__param_{index}__"
                self._param_index_by_name[name] = index
                variables.append(z3.Int(name))
            self._parameters = variables
        return self._parameters

    def state_column(self, values: list[int]) -> Any:
        """A per-state data column (one integer per state, in state order), usable as a z3py Int term."""
        if len(values) != self.num_states:
            raise ValueError(f"state_column expects {self.num_states} values, got {len(values)}")
        _check_fits_int64(values, "state_column")
        column = len(self._state_col_by_name)
        name = f"__state_col_{column}__"
        self._state_col_by_name[name] = column
        for state in range(self.num_states):
            self._state_rows[state].append(int(values[state]))
        return z3.Int(name)

    def choice_column(self, values: list[int]) -> Any:
        """A per-choice data column (one integer per choice, in choice order), usable as a z3py Int term."""
        if len(values) != self.num_choices:
            raise ValueError(f"choice_column expects {self.num_choices} values, got {len(values)}")
        _check_fits_int64(values, "choice_column")
        column = len(self._choice_col_by_name)
        name = f"__choice_col_{column}__"
        self._choice_col_by_name[name] = column
        for choice in range(self.num_choices):
            self._choice_rows[choice].append(int(values[choice]))
        return z3.Int(name)

    @staticmethod
    def in_bits(parameter: Any, data: Any) -> Any:
        """True iff parameter's value is a set bit of data, interpreted as a 64-bit bitmask (one bit per option) -- e.g. a state's or choice's "unavailable
        options" mask.

        parameter must be a bare parameter variable.
        """
        return _IN_BITS_FUNC(parameter, data)

    def share_action(
        self,
        parameter: Any,
        states: Iterable[int],
        choice_to_action: Sequence[int],
        actions: Sequence[int] | None = None,
        disable_others: bool = False,
    ) -> list[int]:
        """Make the given states play the same action: the value of parameter names it, and it enables, in every one of the states, the choice with that action.

        Every choice of these states is colored; all the other choices are left alone. That makes the coloring incomplete unless these are all the states -- the
        rest of the model stays free.

        :param parameter: one of self.parameters
        :param states: the states that play the same action
        :param choice_to_action: the action of every choice of the model, in choice order
        :param actions: the actions, option o of the parameter being actions[o]. By default the actions that every one of the states offers.
        :param disable_others: what to do with a choice whose action is not among actions: color it so that it is never enabled, instead of raising a ValueError
            (which needs the general coloring).
        :returns: actions, i.e. what the options of the parameter stand for: it has to be declared with as many options
        """
        if str(parameter) not in self._param_index_by_name:
            raise ValueError(f"{parameter} is not a parameter of this builder (one of ColoringBuilder.parameters)")
        group = list(states)
        offered = {state: {choice_to_action[choice] for choice in range(self.row_groups[state], self.row_groups[state + 1])} for state in group}
        if actions is None:
            common = set.intersection(*offered.values()) if group else set()
            if not common:
                raise ValueError(f"no action is offered by every one of the states {group}")
            actions = sorted(common)
        actions = list(actions)
        for state in group:
            for action in actions:
                if action not in offered[state]:
                    raise ValueError(f"state {state} does not offer action {action}, so the parameter taking it would leave that state without a choice")
        option_of_action = {action: option for option, action in enumerate(actions)}
        templates = {option: self.template(parameter == option) for option in range(len(actions))}
        never: Template | None = None
        for state in group:
            for choice in range(self.row_groups[state], self.row_groups[state + 1]):
                action = choice_to_action[choice]
                if action in option_of_action:
                    self.color(choice, templates[option_of_action[action]])
                elif disable_others:
                    never = never or self.template(z3.BoolVal(False))
                    self.color(choice, never)
                else:
                    raise ValueError(
                        f"choice {choice} of state {state} has action {action}, which no option of the parameter names: add it to actions, or pass "
                        "disable_others=True to have such choices never enabled"
                    )
        return actions

    def template(self, formula: Any) -> Template:
        """Compile a z3py formula (built over self.parameters / data columns / in_bits) into a reusable Template."""
        return Template(_Walker(self).walk_formula(formula))

    def color(self, choices: int | Iterable[int], template: Template) -> None:
        """Color one choice, or every choice in an iterable of choice indices, with template.

        Every choice may be colored at most once; choices never colored stay uncolored (always enabled), matching Coloring's own convention.
        """
        indices = [choices] if isinstance(choices, int) else list(choices)
        for choice in indices:
            if self._choice_root[choice] is not None:
                raise ValueError(f"choice {choice} is already colored")
            self._choice_root[choice] = template.node

    def build(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace | None = None, general: bool | None = None) -> Any:
        """Produce the coloring: a payntbind.synthesis.Coloring if every color is a conjunction of (parameter == option) pairs, else a ColoringGeneral.

        :param parameter_space: the parameters colors are formulas over. Only needed for the standard coloring, which is sized by the number of options of every
            parameter: without it the coloring is always general.
        :param general: True for a ColoringGeneral whatever the colors are; False for the standard Coloring, raising a ValueError naming the first choice whose
            color cannot be one (or if there is no parameter_space); None (default) to get the standard Coloring whenever it can represent every color.
        """
        if general is True:
            return self._build_general()
        if parameter_space is None:
            if general is False:
                raise ValueError("the standard coloring is sized by the number of options of every parameter: build() needs the parameter_space")
            return self._build_general()
        if parameter_space.num_parameters != self.num_parameters:
            raise ValueError(f"the builder colors over {self.num_parameters} parameters, but the parameter space has {parameter_space.num_parameters}")
        reason = self._not_standard_because_of_relevance()
        pairs: list[list[tuple[int, int]]] | None = None
        if reason is None:
            try:
                pairs = self._choice_to_pairs(parameter_space)
            except _NotStandard as error:
                reason = str(error)
        if pairs is None:
            if general is False:
                raise ValueError(f"the standard coloring cannot represent this coloring: {reason}")
            return self._build_general()
        return payntbind.synthesis.Coloring(parameter_space.native, self.row_groups, pairs)

    def _build_general(self) -> Any:
        import stormpy.storage

        choice_root = [-1 if root is None else root for root in self._choice_root]
        state_is_relevant = self._state_is_relevant
        if state_is_relevant is None:
            state_is_relevant = stormpy.storage.BitVector(0)
        return payntbind.synthesis.ColoringGeneral(
            self.row_groups,
            self.num_parameters,
            self._table.op,
            self._table.a,
            self._table.b,
            self._table.c,
            self._table.option_sets,
            choice_root,
            self._state_rows,
            self._choice_rows,
            state_is_relevant,
        )

    def _not_standard_because_of_relevance(self) -> str | None:
        """The standard coloring has no irrelevant states (ColoringGeneral picks their first choice itself), so any makes it impossible."""
        relevant = self._state_is_relevant
        if relevant is not None and relevant.size() > 0 and relevant.number_of_set_bits() < relevant.size():
            return "some states are irrelevant, which only a general coloring can express"
        return None

    def _choice_to_pairs(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> list[list[tuple[int, int]]]:
        """Every choice's color as a list of (parameter, option) pairs; raises _NotStandard naming the first choice whose color is not such a conjunction, and
        ValueError for an option its parameter does not have."""
        pairs: list[list[tuple[int, int]]] = [[] for _ in range(self.num_choices)]
        # the pairs of a color that reads no data column are the same for every choice it is attached to
        shared: dict[int, list[tuple[int, int]]] = {}
        for state in range(self.num_states):
            for choice in range(self.row_groups[state], self.row_groups[state + 1]):
                root = self._choice_root[choice]
                if root is None:
                    continue
                if root in shared:
                    pairs[choice] = shared[root]
                    continue
                try:
                    reads_data = [False]
                    color = self._conjunction(root, state, choice, reads_data)
                except _NotStandard as error:
                    raise _NotStandard(f"choice {choice} (state {state}): {error}") from None
                for parameter, option in color.items():
                    if not 0 <= option < parameter_space.parameter_num_options(parameter):
                        # not a matter of representation: whatever the coloring, that choice could never be enabled
                        raise ValueError(
                            f"choice {choice} (state {state}) is colored parameter {parameter} == {option}, but parameter {parameter} has only "
                            f"{parameter_space.parameter_num_options(parameter)} options"
                        )
                pairs[choice] = sorted(color.items())
                if not reads_data[0]:
                    shared[root] = pairs[choice]
        return pairs

    def _conjunction(self, node: int, state: int, choice: int, reads_data: list[bool]) -> dict[int, int]:
        """The color at node, for this choice, as {parameter: option} meaning the conjunction of those equalities."""
        table = self._table
        op = table.op[node]
        if op == NodeOp.TrueF:
            return {}
        if op == NodeOp.AndF:
            conjunction = self._conjunction(table.a[node], state, choice, reads_data)
            for parameter, option in self._conjunction(table.b[node], state, choice, reads_data).items():
                if conjunction.setdefault(parameter, option) != option:
                    raise _NotStandard(
                        f"the color contradicts itself (parameter {parameter} is both {conjunction[parameter]} and {option}), so the choice is never enabled"
                    )
            return conjunction
        if op == NodeOp.EqF:
            left, right = table.a[node], table.b[node]
            if table.op[left] == NodeOp.ParamTerm:
                parameter_node, value_node = left, right
            elif table.op[right] == NodeOp.ParamTerm:
                parameter_node, value_node = right, left
            else:
                raise _NotStandard("the color compares two things that are not a parameter")
            return {table.a[parameter_node]: self._data_value(value_node, state, choice, reads_data)}
        if op == NodeOp.InSetF:
            options = table.option_sets[table.b[node]]
            if len(options) != 1:
                raise _NotStandard(f"the color allows {len(options)} options of a parameter, not exactly one")
            return {table.a[table.a[node]]: options[0]}
        raise _NotStandard(f"the color is a {op.name} formula, not a conjunction of (parameter == option) atoms")

    def _data_value(self, node: int, state: int, choice: int, reads_data: list[bool]) -> int:
        """The value of a term that only involves numerals and data columns, at this state and choice."""
        table = self._table
        op = table.op[node]
        if op == NodeOp.ConstTerm:
            return int(table.a[node])
        if op == NodeOp.StateColTerm:
            reads_data[0] = True
            return self._state_rows[state][table.a[node]]
        if op == NodeOp.ChoiceColTerm:
            reads_data[0] = True
            return self._choice_rows[choice][table.a[node]]
        if op in (NodeOp.AddTerm, NodeOp.SubTerm, NodeOp.MulTerm):
            left = self._data_value(table.a[node], state, choice, reads_data)
            right = self._data_value(table.b[node], state, choice, reads_data)
            return left + right if op == NodeOp.AddTerm else left - right if op == NodeOp.SubTerm else left * right
        raise _NotStandard("the color compares a parameter with something that is not fixed by the state and the choice")

    def check_definition(
        self,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        samples: int | None = None,
        seed: int = 0,
        complete: bool = True,
    ) -> None:
        """Check Definition 2 on sampled (or, when samples is None, all) full assignments of parameter_space: every state must have exactly one enabled choice.
        Raises AssertionError on the first violation found. Only practical on small parameter spaces (all_combinations() is exhaustive) or with a modest sample
        size.

        :param parameter_space: built by the caller to match this builder's layout (e.g. reset_tree's own reconstruction from a coloring's parameter_info, for a
            decision-tree coloring) -- the builder itself never holds one, see __init__.
        :param complete: False to accept a coloring that leaves more than one choice enabled in a state (an incomplete coloring), which must still enable at
            least one
        """
        import random

        coloring = self.build(parameter_space)
        combinations = list(parameter_space.all_combinations())
        if samples is not None and samples < len(combinations):
            combinations = random.Random(seed).sample(combinations, samples)
        for combination in combinations:
            assignment = parameter_space.construct_assignment(combination)
            selected = coloring.selectCompatibleChoices(assignment.native)
            for state in range(self.num_states):
                enabled = [choice for choice in range(self.row_groups[state], self.row_groups[state + 1]) if selected[choice]]
                assert len(enabled) == 1 or (not complete and len(enabled) > 1), (
                    f"Definition 2 violated at assignment {combination}, state {state}: {len(enabled)} choices enabled ({enabled}), expected "
                    + ("exactly 1" if complete else "at least 1")
                )
