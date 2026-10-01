"""Unit tests for ColoringBuilder's z3py-to-node-table walker, exercising formula shapes fromColoring never produces (Or, Not, Implies, Ite, in_bits, arithmetic
comparisons) directly against the compiled coloring's selectCompatibleChoices/check_definition, on synthetic row_groups
(selectCompatibleChoices/getStateToHoles/ relevantParameters/areChoicesConsistent only need the coloring and row_groups, not a real transition matrix -- so a
real stormpy model isn't needed here).

End-to-end synthesis on such colorings is exercised via the DT template (see tests/dt), the first real client of these formula shapes.
"""

from __future__ import annotations

import itertools
import random

import pytest
import stormpy.storage
import z3

import paynt.utils.coloring_builder
import paynt.parameter_space.parameter_space

from helpers.helper import load_colored_mdp


def make_parameter_space(domains):
    """A ParameterSpace whose domains are, deliberately, exactly [0..len(domain)-1] for every given domain -- so a parameter's *option index* (what
    holeOptions/parameter_set_options operate on) and the plain integer used in these tests' z3py formulas coincide, keeping the tests' arithmetic legible."""
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for index, domain in enumerate(domains):
        assert list(domain) == list(range(len(domain))), "test domains must be range(len(domain)) -- see docstring"
        parameter_space.add_parameter(f"p{index}", list(domain))
    return parameter_space


class TestOrRecognisesInSet:
    def test_or_of_equalities_is_exact_even_where_a_kleene_or_of_plain_equalities_would_be_unknown(self):
        # 1 state, 2 choices; parameter p in {0,1,2}; choice0 <- Or(p==0, p==1), choice1 <- p==2
        parameter_space = make_parameter_space([[0, 1, 2]])
        row_groups = [0, 2]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = builder.parameters[0]
        builder.color(0, builder.template(z3.Or(p == 0, p == 1)))
        builder.color(1, builder.template(p == 2))
        coloring = builder.build()

        # restrict p to exactly {0,1}: In_set(p,{0,1}) is exactly TRUE here (choice0 certain), and choice1 (p==2) is
        # excluded -- a naive Or(Eq,Eq) translation would instead leave both at Kleene-U (kept, imprecisely)
        subfamily = parameter_space.copy()
        subfamily.parameter_set_options(0, [0, 1])
        selection = coloring.selectCompatibleChoices(subfamily.native)
        assert [selection[choice] for choice in range(2)] == [True, False]


class TestGenericFormulaShapes:
    def test_not_implies_and_ite_match_a_hand_computed_truth_table(self):
        # 1 state, 4 choices, 2 parameters p,q in {0,1}, deliberately routed through Not/Implies/Ite/And so the
        # walker's coverage of those shapes is exercised; each color is logically equivalent to a simple condition
        parameter_space = make_parameter_space([[0, 1], [0, 1]])
        row_groups = [0, 4]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p, q = builder.parameters
        builder.color(0, builder.template(z3.Not(p == 0)))  # p == 1
        builder.color(1, builder.template(z3.Implies(p == 0, q == 1)))  # p==1 or q==1
        builder.color(2, builder.template(z3.If(p == 0, q == 0, q == 1)))  # (p==0 and q==0) or (p==1 and q==1)
        builder.color(3, builder.template(z3.And(p == 0, q == 0)))
        coloring = builder.build()

        for pv, qv in itertools.product((0, 1), (0, 1)):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            assignment.parameter_set_options(1, [qv])
            raw = coloring.selectCompatibleChoices(assignment.native)
            selection = [raw[choice] for choice in range(4)]
            expected = [
                pv == 1,
                pv == 1 or qv == 1,
                (pv == 0 and qv == 0) or (pv == 1 and qv == 1),
                pv == 0 and qv == 0,
            ]
            assert selection == expected, (pv, qv, selection, expected)

    def test_comparisons_and_arithmetic_over_a_data_column(self):
        # 1 state, 2 choices; parameter p in {0,1,2,3}; choice0 <- p + 1 <= column (column == 2, so p in {0,1}),
        # choice1 <- its logical complement, so every assignment has exactly one enabled choice, as a well-formed
        # coloring must (a lone choice0 would leave p in {2,3} with none, tripping ColoringGeneral's "no choice is
        # available" invariant -- the same one ColoringSmt enforces, guaranteed there by the tree's don't-care leaf)
        parameter_space = make_parameter_space([[0, 1, 2, 3]])
        row_groups = [0, 2]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = builder.parameters[0]
        column = builder.state_column([2])
        condition = p + 1 <= column
        builder.color(0, builder.template(condition))
        builder.color(1, builder.template(z3.Not(condition)))
        coloring = builder.build()
        for pv in range(4):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            selection = coloring.selectCompatibleChoices(assignment.native)
            assert selection[0] == (pv <= 1)
            assert selection[1] == (pv > 1)


class TestTermAndComparisonShapes:
    """Every term and comparison the walker accepts, against Python's own arithmetic on every full assignment.

    The truth table above covers addition, `<=`, `==` and `If` as a formula; these are the rest: subtraction, negation, multiplication, `If` as a term and the
    strict and inclusive orders.
    """

    P_OPTIONS = (0, 1, 2)
    Q_OPTIONS = (0, 1, 2)
    COLUMN = 2

    CASES = {
        "subtraction": (lambda p, q, c: p - q >= 1, lambda p, q, c: p - q >= 1),
        "negation": (lambda p, q, c: -p < -q, lambda p, q, c: -p < -q),
        "product": (lambda p, q, c: p * q == 2, lambda p, q, c: p * q == 2),
        "product with a column": (lambda p, q, c: p * c > q, lambda p, q, c: p * c > q),
        "integer if": (lambda p, q, c: z3.If(p == 0, q, c) > 1, lambda p, q, c: (q if p == 0 else c) > 1),
        "nested integer if": (
            lambda p, q, c: z3.If(p < q, z3.If(q == 2, c, p), q - 1) >= 1,
            lambda p, q, c: (c if q == 2 else p) >= 1 if p < q else q - 1 >= 1,
        ),
        "greater or equal": (lambda p, q, c: p >= q, lambda p, q, c: p >= q),
        "greater": (lambda p, q, c: p > q, lambda p, q, c: p > q),
        "less or equal": (lambda p, q, c: p <= q, lambda p, q, c: p <= q),
        "less": (lambda p, q, c: p < q, lambda p, q, c: p < q),
        "an or of equalities over two parameters": (lambda p, q, c: z3.Or(p == 0, q == 1), lambda p, q, c: p == 0 or q == 1),
        "an or of an equality and something else": (lambda p, q, c: z3.Or(p == 0, p + 1 == 2), lambda p, q, c: p == 0 or p + 1 == 2),
    }

    @pytest.mark.parametrize("case", CASES)
    def test_the_selection_on_a_full_assignment_is_the_formula_evaluated(self, case):
        formula, python = self.CASES[case]
        parameter_space = make_parameter_space([list(self.P_OPTIONS), list(self.Q_OPTIONS)])
        # one state: the colored choice, and an uncolored one so that the state keeps a choice whatever the formula says
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], parameter_space.num_parameters)
        p, q = builder.parameters
        column = builder.state_column([self.COLUMN])
        builder.color(0, builder.template(formula(p, q, column)))
        coloring = builder.build()
        for pv, qv in itertools.product(self.P_OPTIONS, self.Q_OPTIONS):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            assignment.parameter_set_options(1, [qv])
            assert coloring.selectCompatibleChoices(assignment.native)[0] == python(pv, qv, self.COLUMN), (pv, qv)

    def test_equal_formulas_share_one_node(self):
        """The option set of an Or of equalities is a set, so the order (and the repetition) of the options does not matter."""
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 3], 2)
        p, q = builder.parameters
        template = builder.template(z3.Or(p == 0, p == 2))
        assert builder.template(z3.Or(p == 2, p == 0)).node == template.node
        assert builder.template(z3.Or(p == 2, p == 0, p == 2)).node == template.node
        assert builder.template(z3.Or(q == 0, q == 2)).node != template.node


class TestFormulaErrors:
    @staticmethod
    def builder():
        return paynt.utils.coloring_builder.ColoringBuilder([0, 2, 4], 2)

    def test_a_quantifier_is_not_a_formula_of_the_language(self):
        builder = self.builder()
        x = z3.Int("x")
        with pytest.raises(ValueError, match="unsupported formula in a general coloring template"):
            builder.template(z3.ForAll([x], x > 0))

    def test_an_unsupported_comparison_is_named(self):
        builder = self.builder()
        p, q = builder.parameters
        with pytest.raises(ValueError, match="unsupported term in a general coloring template"):
            builder.template(p / q > 0)

    def test_a_connective_outside_the_language_is_refused(self):
        builder = self.builder()
        p, q = builder.parameters
        with pytest.raises(ValueError, match="unsupported formula in a general coloring template"):
            builder.template(z3.Xor(p == 0, q == 0))

    def test_in_bits_needs_a_bare_parameter(self):
        builder = self.builder()
        p, _q = builder.parameters
        mask = builder.state_column([1, 1])
        with pytest.raises(ValueError, match="in_bits' first argument must be a bare parameter reference"):
            builder.template(builder.in_bits(p + 1, mask))

    def test_distinct_of_more_than_two_terms_is_refused(self):
        builder = self.builder()
        p, q = builder.parameters
        with pytest.raises(ValueError, match="Distinct is only supported with exactly 2 arguments"):
            builder.template(z3.Distinct(p, q, builder.state_column([0, 0])))

    def test_a_column_has_to_have_a_value_for_every_state_and_every_choice(self):
        builder = self.builder()
        with pytest.raises(ValueError, match="state_column expects 2 values, got 3"):
            builder.state_column([0, 1, 2])
        with pytest.raises(ValueError, match="choice_column expects 4 values, got 1"):
            builder.choice_column([0])

    def test_a_refused_formula_leaves_the_builder_usable(self):
        builder = self.builder()
        p, q = builder.parameters
        with pytest.raises(ValueError):
            builder.template(p / q > 0)
        builder.color(0, builder.template(p == 1))
        assert kind_of(builder.build(make_parameter_space([[0, 1], [0, 1]]))) == "Coloring"


class TestWhyAColorIsNotStandard:
    """Why build(general=False) refuses: what the standard coloring cannot represent, in a color that is otherwise fine for the general one."""

    @staticmethod
    def builder_with(formula):
        parameter_space = make_parameter_space([[0, 1, 2]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], 1)
        flag = builder.state_column([1])
        builder.color(0, builder.template(formula(builder.parameters[0], flag)))
        return builder, parameter_space

    def test_a_comparison_of_two_things_that_are_not_a_parameter(self):
        builder, parameter_space = self.builder_with(lambda p, flag: flag == 1)
        with pytest.raises(ValueError, match="compares two things that are not a parameter"):
            builder.build(parameter_space, general=False)
        assert kind_of(builder.build(parameter_space)) == "ColoringGeneral"

    def test_an_option_set_of_several_options(self):
        builder, parameter_space = self.builder_with(lambda p, flag: z3.Or(p == 0, p == 1))
        with pytest.raises(ValueError, match="allows 2 options of a parameter, not exactly one"):
            builder.build(parameter_space, general=False)

    def test_a_comparison_with_something_the_state_and_choice_do_not_fix(self):
        builder, parameter_space = self.builder_with(lambda p, flag: p == z3.If(flag == 1, 2, 0))
        with pytest.raises(ValueError, match="not fixed by the state and the choice"):
            builder.build(parameter_space, general=False)

    def test_an_option_set_of_one_option_is_a_pair_like_any_other(self):
        """Z3 keeps the repeated option in an Or, so this is the one-option set the walker builds; as a pair it is the standard coloring's."""
        builder, parameter_space = self.builder_with(lambda p, flag: z3.Or(p == 1, p == 1))
        assert kind_of(builder.build(parameter_space)) == "Coloring"
        assert builder.build(parameter_space).getChoiceToAssignment()[0] == [(0, 1)]


class TestCheckDefinitionOnASample:
    """A coloring that violates Definition 2 at a single assignment of nine: only a sample that happens to draw it notices."""

    @staticmethod
    def broken_at_one_assignment():
        parameter_space = make_parameter_space([[0, 1, 2], [0, 1, 2]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], 2)
        p, q = builder.parameters
        # choice 0 is uncolored (always enabled), so choice 1 must never be: it is, at p == q == 2
        builder.color(1, builder.template(z3.And(p == 2, q == 2)))
        return builder, parameter_space

    @staticmethod
    def noticed(builder, parameter_space, **kwargs):
        try:
            builder.check_definition(parameter_space, **kwargs)
        except AssertionError as error:
            assert "Definition 2 violated at assignment (2, 2)" in str(error)
            return True
        return False

    def test_the_whole_space_notices_it(self):
        builder, parameter_space = self.broken_at_one_assignment()
        assert self.noticed(builder, parameter_space)
        assert self.noticed(builder, parameter_space, samples=9)
        assert self.noticed(builder, parameter_space, samples=100)  # more than the space has: all of it

    def test_a_smaller_sample_may_miss_it_and_a_seed_always_draws_the_same_one(self):
        builder, parameter_space = self.broken_at_one_assignment()
        outcomes = [self.noticed(builder, parameter_space, samples=3, seed=seed) for seed in range(20)]
        assert outcomes == [self.noticed(builder, parameter_space, samples=3, seed=seed) for seed in range(20)]
        # 3 of the 9 assignments are drawn: the bad one is among them for a third of the seeds, so over twenty both outcomes occur
        assert True in outcomes
        assert False in outcomes


class TestInBits:
    def test_in_bits_matches_a_hand_computed_bitmask(self):
        # 1 state, 2 choices; parameter p in {0,1,2,3}; the state's data column is a bitmask with bits 0 and 2 set
        # (0b0101 = 5); choice0 <- p's value is one of those set bits, choice1 <- its complement (see above)
        parameter_space = make_parameter_space([[0, 1, 2, 3]])
        row_groups = [0, 2]
        builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = builder.parameters[0]
        mask = builder.state_column([5])
        condition = builder.in_bits(p, mask)
        builder.color(0, builder.template(condition))
        builder.color(1, builder.template(z3.Not(condition)))
        coloring = builder.build()
        for pv in range(4):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [pv])
            selection = coloring.selectCompatibleChoices(assignment.native)
            assert selection[0] == (pv in (0, 2))
            assert selection[1] == (pv not in (0, 2))


class TestCheckDefinition:
    def test_passes_for_a_well_formed_coloring_and_raises_for_a_broken_one(self):
        parameter_space = make_parameter_space([[0, 1]])
        row_groups = [0, 2]

        good = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = good.parameters[0]
        good.color(0, good.template(p == 0))
        good.color(1, good.template(p == 1))
        good.check_definition(parameter_space)

        broken = paynt.utils.coloring_builder.ColoringBuilder(row_groups, parameter_space.num_parameters)
        p = broken.parameters[0]
        broken.color(0, broken.template(p == 0))
        broken.color(1, broken.template(p == 0))  # both choices share the same color: 2 enabled at p=0, 0 at p=1
        with pytest.raises(AssertionError):
            broken.check_definition(parameter_space)


class TestColoringErrors:
    def test_coloring_the_same_choice_twice_raises(self):
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1], parameter_space.num_parameters)
        p = builder.parameters[0]
        builder.color(0, builder.template(p == 0))
        with pytest.raises(ValueError):
            builder.color(0, builder.template(p == 1))

    def test_an_unsupported_formula_shape_raises_value_error(self):
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1], parameter_space.num_parameters)
        p = builder.parameters[0]
        x = z3.Real("x")  # a sort this DSL doesn't support at all
        with pytest.raises(ValueError):
            builder.template(z3.And(p == 0, x > 0))


class TestDataColumnRange:
    """A data column holds signed 64-bit integers: anything wider is refused with a clear error, instead of pybind's opaque TypeError at build()."""

    @pytest.mark.parametrize("value", [2**63, -(2**63) - 1, 2**64, 1 << 70])
    def test_a_value_that_does_not_fit_is_refused(self, value):
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1, 2], 1)
        with pytest.raises(ValueError, match=r"state_column value .* at index 1 does not fit in a signed 64-bit integer"):
            builder.state_column([0, value])
        with pytest.raises(ValueError, match=r"choice_column value .* at index 0 does not fit in a signed 64-bit integer"):
            builder.choice_column([value, 0])

    def test_the_extreme_values_that_do_fit_are_accepted(self):
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], 1)
        p = builder.parameters[0]
        highest = builder.state_column([2**63 - 1])
        lowest = builder.choice_column([-(2**63), 0])
        builder.color(0, builder.template(z3.And(p == 0, lowest < highest)))
        builder.color(1, builder.template(p == 1))
        builder.build()

    def test_a_refused_column_leaves_no_trace_in_the_builder(self):
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], 1)
        p = builder.parameters[0]
        with pytest.raises(ValueError):
            builder.state_column([2**63])
        x = builder.state_column([1])  # the first column that exists: had the refused one been registered, x would name a column of another state's data
        builder.color(0, builder.template(p == x))
        builder.color(1, builder.template(p != x))
        coloring = builder.build()
        family = parameter_space.copy()
        family.parameter_set_options(0, [1])
        selection = coloring.selectCompatibleChoices(family.native)
        assert [selection[choice] for choice in range(2)] == [True, False]


def kind_of(coloring):
    return type(coloring).__name__


class TestStandardColoring:
    """build(parameter_space) gives the standard Coloring whenever every color is a conjunction of (parameter == option) pairs, the general one otherwise."""

    @staticmethod
    def two_states(colors, domains=((0, 1, 2), (0, 1))):
        """A builder over 2 states of 2 choices each and 2 parameters, colored by colors: {choice: formula over (p, q, builder)}; the choices not in colors are
        left uncolored."""
        parameter_space = make_parameter_space([list(domain) for domain in domains])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2, 4], parameter_space.num_parameters)
        p, q = builder.parameters
        for choice, formula in colors.items():
            builder.color(choice, builder.template(formula(p, q, builder)))
        return builder, parameter_space

    def test_a_conjunction_of_equalities_is_the_standard_coloring(self):
        builder, parameter_space = self.two_states({0: lambda p, q, b: z3.And(p == 1, q == 0), 1: lambda p, q, b: p == 2, 3: lambda p, q, b: q == 1})
        coloring = builder.build(parameter_space)
        assert kind_of(coloring) == "Coloring"
        assert [sorted(pairs) for pairs in coloring.getChoiceToAssignment()] == [[(0, 1), (1, 0)], [(0, 2)], [], [(1, 1)]]  # choice 2 is uncolored

    def test_the_operands_of_an_equality_may_come_in_either_order_and_repeat(self):
        builder, parameter_space = self.two_states({0: lambda p, q, b: z3.And(z3.IntVal(1) == p, p == 1, q == 0, z3.BoolVal(True))})
        assert builder.build(parameter_space).getChoiceToAssignment()[0] == [(0, 1), (1, 0)]

    def test_data_columns_are_read_at_every_choice(self):
        parameter_space = make_parameter_space([[0, 1, 2], [0, 1, 2]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2, 4], 2)
        p, q = builder.parameters
        action = builder.choice_column([0, 2, 1, 1])
        state = builder.state_column([1, 0])
        for choice in range(4):
            builder.color(choice, builder.template(z3.And(p == action, q == state + 1)))
        assert builder.build(parameter_space).getChoiceToAssignment() == [[(0, 0), (1, 2)], [(0, 2), (1, 2)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]]

    def test_one_template_colors_many_choices(self):
        parameter_space = make_parameter_space([[0, 1], [0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2, 4], 2)
        template = builder.template(builder.parameters[0] == 1)
        builder.color([0, 1, 3], template)
        assert builder.build(parameter_space).getChoiceToAssignment() == [[(0, 1)], [(0, 1)], [], [(0, 1)]]

    @pytest.mark.parametrize(
        "formula",
        [
            lambda p, q, b: z3.Or(p == 0, p == 2),
            lambda p, q, b: p != 1,
            lambda p, q, b: p <= 1,
            lambda p, q, b: z3.Not(p == 1),
            lambda p, q, b: z3.BoolVal(False),
            lambda p, q, b: z3.And(p == 0, p == 1),  # contradicts itself: never enabled
            lambda p, q, b: p == q,
            lambda p, q, b: z3.Implies(q == 0, p == 1),
        ],
        ids=["or", "distinct", "order", "not", "false", "contradiction", "two parameters", "implication"],
    )
    def test_any_other_color_needs_the_general_coloring(self, formula):
        builder, parameter_space = self.two_states({1: formula})
        assert kind_of(builder.build(parameter_space)) == "ColoringGeneral"
        with pytest.raises(ValueError, match=r"cannot represent this coloring: choice 1 \(state 0\)"):
            builder.build(parameter_space, general=False)

    def test_the_general_coloring_can_be_asked_for(self):
        builder, parameter_space = self.two_states({0: lambda p, q, b: p == 1})
        assert kind_of(builder.build(parameter_space, general=True)) == "ColoringGeneral"

    def test_without_a_parameter_space_the_coloring_is_general(self):
        """The decision tree template, which determines its parameters itself, builds this way."""
        builder, _parameter_space = self.two_states({0: lambda p, q, b: p == 1})
        assert kind_of(builder.build()) == "ColoringGeneral"
        with pytest.raises(ValueError, match="needs the parameter_space"):
            builder.build(general=False)

    def test_an_option_the_parameter_does_not_have_is_an_error(self):
        builder, parameter_space = self.two_states({0: lambda p, q, b: p == 5})
        with pytest.raises(ValueError, match=r"choice 0 \(state 0\) is colored parameter 0 == 5, but parameter 0 has only 3 options"):
            builder.build(parameter_space)

    def test_a_parameter_space_of_another_size_is_an_error(self):
        builder, _parameter_space = self.two_states({})
        with pytest.raises(ValueError, match="2 parameters, but the parameter space has 1"):
            builder.build(make_parameter_space([[0, 1]]))

    def test_irrelevant_states_need_the_general_coloring(self):
        parameter_space = make_parameter_space([[0, 1]])
        relevant = stormpy.storage.BitVector(2, True)
        relevant.set(1, False)
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1, 2], 1, state_is_relevant=relevant)
        builder.color(0, builder.template(builder.parameters[0] == 0))
        assert kind_of(builder.build(parameter_space)) == "ColoringGeneral"
        with pytest.raises(ValueError, match="irrelevant"):
            builder.build(parameter_space, general=False)

    def test_both_colorings_select_the_same_choices(self):
        # every family enables something in both states: state 0 is complete (p == 0 or p == 1), state 1 always has its uncolored choice 3
        builder, parameter_space = self.two_states(
            {0: lambda p, q, b: p == 0, 1: lambda p, q, b: p == 1, 2: lambda p, q, b: z3.And(p == 1, q == 1)}, domains=((0, 1), (0, 1))
        )
        standard = builder.build(parameter_space, general=False)
        general = builder.build(parameter_space, general=True)
        rng = random.Random(0)
        for _ in range(30):
            family = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                family.parameter_set_options(parameter, sorted(rng.sample(options, rng.randint(1, len(options)))))
            assert set(standard.selectCompatibleChoices(family.native)) == set(general.selectCompatibleChoices(family.native))


@pytest.mark.parametrize("project", ["tests/smpmc-tiny", "tests/mdp-family-avoid-8-2-easy"])
class TestBuilderReproducesTheFixturesColorings:
    """Every choice of a real fixture colored, through the builder, with the conjunction it already has: the builder must hand back an equal standard Coloring
    -- and a general one selecting the same choices."""

    @staticmethod
    def rebuilt(project):
        colored_mdp, _task = load_colored_mdp(project)
        parameter_space = colored_mdp.parameter_space
        pairs = colored_mdp.coloring.getChoiceToAssignment()
        builder = paynt.utils.coloring_builder.ColoringBuilder(colored_mdp.underlying_mdp.nondeterministic_choice_indices, parameter_space.num_parameters)
        for choice, choice_pairs in enumerate(pairs):
            if choice_pairs:
                builder.color(choice, builder.template(z3.And(*[builder.parameters[parameter] == option for parameter, option in choice_pairs])))
        return colored_mdp, builder

    def test_the_standard_coloring_is_equal(self, project):
        colored_mdp, builder = self.rebuilt(project)
        standard = builder.build(colored_mdp.parameter_space)
        assert kind_of(standard) == "Coloring"
        assert [sorted(pairs) for pairs in standard.getChoiceToAssignment()] == [sorted(pairs) for pairs in colored_mdp.coloring.getChoiceToAssignment()]
        assert [set(holes) for holes in standard.getStateToHoles()] == [set(holes) for holes in colored_mdp.coloring.getStateToHoles()]

    def test_it_selects_the_same_choices_as_the_general_coloring_too(self, project):
        colored_mdp, builder = self.rebuilt(project)
        parameter_space = colored_mdp.parameter_space
        standard = builder.build(parameter_space)
        general = builder.build(parameter_space, general=True)
        rng = random.Random(1)
        for _ in range(20):
            family = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                if len(options) > 1 and rng.random() < 0.7:
                    family.parameter_set_options(parameter, sorted(rng.sample(options, rng.randint(1, len(options)))))
            expected = set(colored_mdp.coloring.selectCompatibleChoices(family.native))
            assert set(standard.selectCompatibleChoices(family.native)) == expected
            assert set(general.selectCompatibleChoices(family.native)) == expected


class TestCheckDefinitionOfAnIncompleteColoring:
    """A state whose choices are not exclusive: only an incomplete coloring may have one."""

    @staticmethod
    def builder_with_two_uncolored_choices():
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], 1)
        builder.color(0, builder.template(builder.parameters[0] == 0))  # choice 1 stays uncolored: always enabled next to it
        return builder, parameter_space

    def test_a_state_with_several_enabled_choices_violates_definition_2(self):
        builder, parameter_space = self.builder_with_two_uncolored_choices()
        with pytest.raises(AssertionError, match="expected exactly 1"):
            builder.check_definition(parameter_space)

    def test_it_is_accepted_as_an_incomplete_coloring(self):
        builder, parameter_space = self.builder_with_two_uncolored_choices()
        builder.check_definition(parameter_space, complete=False)

    def test_a_state_with_no_enabled_choice_is_never_accepted(self):
        parameter_space = make_parameter_space([[0, 1]])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1], 1)
        builder.color(0, builder.template(builder.parameters[0] == 0))
        with pytest.raises(AssertionError, match="at least 1"):
            builder.check_definition(parameter_space, complete=False)


class TestShareAction:
    """builder.share_action(parameter, states, choice_to_action): the states play the same action, the one the parameter's value names."""

    # states 0 and 1 offer the actions 0, 1, 2; state 2 only 0 and 1
    ROW_GROUPS = [0, 3, 6, 8]
    CHOICE_TO_ACTION = [0, 1, 2, 0, 1, 2, 0, 1]

    def builder(self):
        return paynt.utils.coloring_builder.ColoringBuilder(self.ROW_GROUPS, 1)

    def test_the_states_take_the_action_of_the_parameter(self):
        builder = self.builder()
        assert builder.share_action(builder.parameters[0], [0, 1], self.CHOICE_TO_ACTION) == [0, 1, 2]
        parameter_space = make_parameter_space([[0, 1, 2]])
        coloring = builder.build(parameter_space)
        assert kind_of(coloring) == "Coloring"
        assert coloring.getChoiceToAssignment() == [[(0, 0)], [(0, 1)], [(0, 2)], [(0, 0)], [(0, 1)], [(0, 2)], [], []]  # state 2 is not in the group
        assignment = parameter_space.copy()
        assignment.parameter_set_options(0, [1])
        assert set(coloring.selectCompatibleChoices(assignment.native)) == {1, 4, 6, 7}  # the shared action in both states, all of the others' choices

    def test_the_options_are_the_actions_every_state_offers(self):
        builder = self.builder()
        assert builder.share_action(builder.parameters[0], [0, 2], self.CHOICE_TO_ACTION, disable_others=True) == [0, 1]  # state 0 also offers action 2

    def test_a_choice_no_option_names_is_an_error_unless_it_is_disabled(self):
        builder = self.builder()
        with pytest.raises(ValueError, match="choice 2 of state 0 has action 2, which no option of the parameter names"):
            builder.share_action(builder.parameters[0], [0, 2], self.CHOICE_TO_ACTION)

        builder = self.builder()
        builder.share_action(builder.parameters[0], [0, 2], self.CHOICE_TO_ACTION, disable_others=True)
        parameter_space = make_parameter_space([[0, 1]])
        coloring = builder.build(parameter_space)
        assert kind_of(coloring) == "ColoringGeneral"  # a choice that is never enabled is not a conjunction of atoms
        for option in (0, 1):
            assignment = parameter_space.copy()
            assignment.parameter_set_options(0, [option])
            assert not coloring.selectCompatibleChoices(assignment.native)[2]

    def test_the_actions_can_be_given(self):
        builder = self.builder()
        assert builder.share_action(builder.parameters[0], [0, 1], self.CHOICE_TO_ACTION, actions=[2, 0, 1]) == [2, 0, 1]
        assert builder.build(make_parameter_space([[0, 1, 2]])).getChoiceToAssignment()[:3] == [[(0, 1)], [(0, 2)], [(0, 0)]]  # option 0 is action 2

    def test_an_action_a_state_does_not_offer_is_an_error(self):
        """That option would leave the state without a choice."""
        builder = self.builder()
        with pytest.raises(ValueError, match="state 2 does not offer action 2"):
            builder.share_action(builder.parameters[0], [0, 1, 2], self.CHOICE_TO_ACTION, actions=[0, 1, 2])

    def test_states_without_a_common_action_are_an_error(self):
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 1, 2], 1)
        with pytest.raises(ValueError, match="no action is offered by every one of the states"):
            builder.share_action(builder.parameters[0], [0, 1], [0, 1])

    def test_the_parameter_must_be_one_of_the_builders(self):
        builder = self.builder()
        with pytest.raises(ValueError, match="not a parameter of this builder"):
            builder.share_action(z3.Int("other"), [0], self.CHOICE_TO_ACTION)

    def test_several_groups_take_a_parameter_each(self):
        builder = paynt.utils.coloring_builder.ColoringBuilder(self.ROW_GROUPS, 2)
        first, second = builder.parameters
        builder.share_action(first, [0], self.CHOICE_TO_ACTION)
        builder.share_action(second, [1, 2], self.CHOICE_TO_ACTION, disable_others=True)
        parameter_space = make_parameter_space([[0, 1, 2], [0, 1]])
        builder.check_definition(parameter_space)
