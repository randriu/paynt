"""choice_formula: the color of a choice of a general coloring, read back as a z3py formula over the parameters, which are named as in the parameter space.

The formula of a choice has to hold under a full parameter assignment exactly where selectCompatibleChoices enables the choice, so that is what is checked, on
random formulas of the builder's language, on the data columns and the in_bits atoms it substitutes, on irrelevant states and states without choices, on the
decision-tree colorings (the main client of general colorings) and on standard colorings lifted into general ones. The parameters of every test have names of
their own, never p0, p1, ..., and the tests of the names are not written with the function that makes them.
"""

from __future__ import annotations

import bisect
import itertools
import random
import sys

import pytest
import stormpy
import z3
import z3.z3util

import payntbind.synthesis
import paynt.parameter_space.parameter_space
import paynt.utils.coloring
import paynt.utils.coloring_builder

from tests.helpers.formulas import FormulaGenerator
from tests.helpers.models import general_colored_mdp, load_colored_mdp, load_sketch

choice_formula = paynt.utils.coloring.choice_formula
parameter_variables = paynt.utils.coloring.parameter_variables
parameter_space_formula = paynt.utils.coloring.parameter_space_formula
ColoringBuilder = paynt.utils.coloring_builder.ColoringBuilder
NodeOp = payntbind.synthesis.ColoringGeneralNodeOp


def holds(formula, values, parameter_space):
    """Whether the assignment (an option index for every parameter) satisfies the formula."""
    decided = z3.simplify(z3.substitute(formula, *zip(parameter_variables(parameter_space), map(z3.IntVal, values), strict=True)))
    assert z3.is_true(decided) or z3.is_false(decided), f"a full assignment does not decide {formula}: {decided}"
    return z3.is_true(decided)


def equivalent(left, right):
    solver = z3.Solver()
    solver.add(z3.Xor(left, right))
    return solver.check() == z3.unsat


def names_in(formula):
    return {str(variable) for variable in z3.z3util.get_vars(formula)}


def parameter_space_of(domains, names):
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for name, domain in zip(names, domains, strict=True):
        parameter_space.add_parameter(name, domain)
    return parameter_space


def enabled(coloring, parameter_space, values):
    return coloring.selectCompatibleChoices(parameter_space.construct_assignment(values).native)


def choice_mask(num_choices, choices):
    mask = stormpy.datastructures.BitVector(num_choices, False)
    for choice in choices:
        mask.set(choice, True)
    return mask


def take_together(coloring, parameter_space, choices):
    """Whether some assignment of the parameter space enables all the choices, asked of z3 through their formulas: the option of every parameter in such an
    assignment, or None."""
    solver = z3.Solver()
    solver.add(parameter_space_formula(parameter_space), *(choice_formula(coloring, choice, parameter_space) for choice in choices))
    if solver.check() != z3.sat:
        return None
    model = solver.model()
    return [model.eval(variable, model_completion=True).as_long() for variable in parameter_variables(parameter_space)]


RANDOM_NUM_FORMULAS = 40
RANDOM_DOMAINS = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
RANDOM_NAMES = ["mode", "threshold", "action"]


def random_coloring(seed):
    """One state with RANDOM_NUM_FORMULAS colored choices (random formulas over three parameters, a data column and a bit mask) and an uncolored one: the
    parameter space, named RANDOM_NAMES, and the coloring."""
    rng = random.Random(seed)
    builder = ColoringBuilder([0, RANDOM_NUM_FORMULAS + 1], len(RANDOM_DOMAINS))
    column, mask = builder.state_column([rng.randint(-1, 3)]), builder.state_column([rng.randint(0, 2**8 - 1)])
    generator = FormulaGenerator(rng, builder.parameters, column, mask)
    for choice in range(RANDOM_NUM_FORMULAS):
        expression, _function = generator.boolean(rng.randint(1, 3))
        builder.color(choice, builder.template(expression))
    return parameter_space_of(RANDOM_DOMAINS, RANDOM_NAMES), builder.build()


class TestTheAccessors:
    """What choice_formula reads from the coloring."""

    def test_the_node_table_comes_back_as_it_was_given(self):
        # p == 2, p in {0, 2}, and the choices of one state colored with them and left uncolored
        ops = [NodeOp.ParamTerm, NodeOp.ConstTerm, NodeOp.EqF, NodeOp.InSetF]
        node_a, node_b, node_c = [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]
        option_sets = [[0, 2]]
        coloring = payntbind.synthesis.ColoringGeneral(
            [0, 3], 1, ops, node_a, node_b, node_c, option_sets, [2, 3, -1], [[]], [[], [], []], stormpy.datastructures.BitVector(0)
        )
        assert coloring.getNodeTable() == ([int(op) for op in ops], node_a, node_b, node_c, option_sets)
        assert [coloring.getChoiceColor(choice)[0] for choice in range(3)] == [2, 3, -1]

    def test_what_the_color_of_a_choice_is_made_of(self):
        """Seven choices in four states, the second of which has none, and the second and the fourth of which are irrelevant."""
        builder = ColoringBuilder([0, 2, 2, 4, 7], 1, state_is_relevant=TestUncoloredAndIrrelevantStates.relevance())
        builder.state_column([0, 7, 1, 9])
        builder.choice_column([10, 11, 12, 13, 14, 15, 16])
        (p,) = builder.parameters
        builder.color([0, 3, 5], builder.template(p == 1))
        coloring = builder.build()
        states = [0, 0, 2, 2, 3, 3, 3]
        for choice in range(7):
            root, state_row, choice_row, state_is_relevant, is_first_of_state = coloring.getChoiceColor(choice)
            assert (root >= 0) == (choice in (0, 3, 5)), choice
            assert state_row == [[0, 7, 1, 9][states[choice]]], choice
            assert choice_row == [10 + choice], choice
            assert state_is_relevant == (states[choice] in (0, 2)), choice
            assert is_first_of_state == (choice in (0, 2, 4)), choice


class TestParameterVariables:
    def test_the_variables_are_named_as_the_parameters(self):
        parameter_space = parameter_space_of([[0, 1], [0, 1, 2], [0]], ["alpha", "beta", "gamma"])
        variables = parameter_variables(parameter_space)
        assert [str(variable) for variable in variables] == ["alpha", "beta", "gamma"]
        assert all(variable.sort() == z3.IntSort() for variable in variables)

    def test_two_parameters_that_share_a_name_are_not_one_variable(self):
        """Z3 identifies a variable by its name, so every variable is then named with the index of its parameter, as in the BitVec encoding of the engines."""
        parameter_space = parameter_space_of([[0, 1, 2], [0, 1], [0, 1, 2]], ["x", "y", "x"])
        assert [str(variable) for variable in parameter_variables(parameter_space)] == ["x#0", "y#1", "x#2"]
        builder = ColoringBuilder([0, 2], 3)
        first, _second, third = builder.parameters
        builder.color(0, builder.template(z3.And(first == 1, third == 2)))
        formula = choice_formula(builder.build(), 0, parameter_space)
        assert names_in(formula) == {"x#0", "x#2"}
        assert equivalent(formula, z3.And(z3.Int("x#0") == 1, z3.Int("x#2") == 2))

    def test_the_names_are_never_the_same_for_two_parameters(self):
        """Not even where a name already looks like another's with an index, which is why the index is the last thing after the last hash."""
        parameter_space = parameter_space_of([[0, 1]] * 3, ["a", "a", "a#1"])
        names = [str(variable) for variable in parameter_variables(parameter_space)]
        assert names == ["a#0", "a#1", "a#1#2"]

    def test_the_names_stay_as_they_are_where_they_are_all_different(self):
        parameter_space = parameter_space_of([[0, 1]] * 3, ["a", "a#1", "b"])
        assert [str(variable) for variable in parameter_variables(parameter_space)] == ["a", "a#1", "b"]

    def test_a_parameter_space_without_parameters(self):
        assert parameter_variables(parameter_space_of([], [])) == []


@pytest.mark.parametrize("seed", range(6))
class TestRandomFormulas:
    def test_the_formula_of_a_choice_holds_exactly_where_the_choice_is_enabled(self, seed):
        parameter_space, coloring = random_coloring(seed)
        formulas = [choice_formula(coloring, choice, parameter_space) for choice in range(RANDOM_NUM_FORMULAS + 1)]
        for values in itertools.product(*RANDOM_DOMAINS):
            selected = enabled(coloring, parameter_space, values)
            for choice, formula in enumerate(formulas):
                assert holds(formula, values, parameter_space) == bool(selected[choice]), (choice, values, formula)

    def test_the_formulas_are_over_the_names_of_the_parameters(self, seed):
        parameter_space, coloring = random_coloring(seed)
        used = set()
        for choice in range(RANDOM_NUM_FORMULAS + 1):
            used |= names_in(choice_formula(coloring, choice, parameter_space))
        assert used == set(RANDOM_NAMES)

    def test_without_a_parameter_space_the_variables_are_numbered_and_the_formulas_are_the_same(self, seed):
        parameter_space, coloring = random_coloring(seed)
        numbered = [z3.Int(f"p{index}") for index in range(len(RANDOM_NAMES))]
        used = set()
        for choice in range(RANDOM_NUM_FORMULAS + 1):
            formula = choice_formula(coloring, choice)
            used |= names_in(formula)
            named = choice_formula(coloring, choice, parameter_space)
            assert equivalent(formula, z3.substitute(named, *zip(parameter_variables(parameter_space), numbered, strict=True))), choice
        assert used == {"p0", "p1", "p2"}


class TestParameterSpaceFormula:
    def test_every_parameter_takes_one_of_its_options(self):
        parameter_space = parameter_space_of([[0, 1, 2], [0], list(range(5))], ["a", "b", "c"])
        a, b, c = z3.Int("a"), z3.Int("b"), z3.Int("c")
        assert equivalent(parameter_space_formula(parameter_space), z3.And(z3.Or(a == 0, a == 1, a == 2), b == 0, c >= 0, c <= 4))

    def test_a_family_has_the_options_it_is_restricted_to(self):
        parameter_space = parameter_space_of([list(range(8)), list(range(3))], ["a", "b"])
        family = parameter_space.copy()
        family.parameter_set_options(0, [0, 1, 4, 6, 7])
        family.parameter_set_options(1, [2])
        formula = parameter_space_formula(family)
        for a, b in itertools.product(range(-1, 9), range(-1, 4)):
            assert holds(formula, (a, b), family) == (a in (0, 1, 4, 6, 7) and b == 2), (a, b)
        assert holds(parameter_space_formula(parameter_space), (3, 1), parameter_space), "the whole space has the options that the family has left out"

    def test_a_long_range_of_options_is_a_range_and_not_a_disjunction(self):
        assert len(str(parameter_space_formula(parameter_space_of([list(range(1000))], ["a"])))) < 40

    def test_the_variables_are_those_of_parameter_variables(self):
        assert names_in(parameter_space_formula(parameter_space_of([[0, 1], [0, 1]], ["x", "x"]))) == {"x#0", "x#1"}

    def test_a_space_without_parameters_and_a_parameter_without_options(self):
        assert z3.is_true(parameter_space_formula(parameter_space_of([], [])))
        family = parameter_space_of([[0, 1], [0, 1]], ["a", "b"]).copy()
        family.parameter_set_options(1, [])
        assert equivalent(parameter_space_formula(family), z3.BoolVal(False))


@pytest.mark.parametrize("seed", range(6))
class TestChoicesTakenTogetherOnRandomColorings:
    """Whether some assignment enables a set of choices at once: z3 answers it about their formulas and the parameter space, and the engine in C++ answers it
    with areChoicesConsistent.

    They must agree, and what z3 finds must enable the choices.
    """

    def test_the_verdict_agrees_with_the_engine_and_the_assignment_enables_the_choices(self, seed):
        parameter_space, coloring = random_coloring(seed)
        num_choices = RANDOM_NUM_FORMULAS + 1
        rng = random.Random(seed)
        verdicts = set()
        for _ in range(30):
            choices = rng.sample(range(num_choices), rng.randint(1, 4))
            consistent, _selection = coloring.areChoicesConsistent(choice_mask(num_choices, choices), parameter_space.native)
            values = take_together(coloring, parameter_space, choices)
            assert (values is not None) == consistent, choices
            verdicts.add(consistent)
            if values is not None:
                assert all(enabled(coloring, parameter_space, values)[choice] for choice in choices), (choices, values)
            else:
                for candidate in itertools.product(*RANDOM_DOMAINS):  # the engine is the judge of each of the assignments
                    assert not all(enabled(coloring, parameter_space, candidate)[choice] for choice in choices), (choices, candidate)
        assert verdicts == {True, False}

    def test_the_assignment_is_one_of_the_options_that_a_family_has_left(self, seed):
        parameter_space, coloring = random_coloring(seed)
        num_choices = RANDOM_NUM_FORMULAS + 1
        rng = random.Random(seed + 100)
        for _ in range(30):
            family = parameter_space.copy()
            for parameter, domain in enumerate(RANDOM_DOMAINS):
                family.parameter_set_options(parameter, sorted(rng.sample(domain, rng.randint(1, 3))))
            choices = rng.sample(range(num_choices), rng.randint(1, 3))
            consistent, _selection = coloring.areChoicesConsistent(choice_mask(num_choices, choices), family.native)
            values = take_together(coloring, family, choices)
            assert (values is not None) == consistent, (choices, str(family))
            if values is not None:
                assert all(value in family.parameter_options(parameter) for parameter, value in enumerate(values)), (values, str(family))


@pytest.mark.parametrize(("project", "depth"), [("tests/dt-maze", 0), ("tests/dt-maze", 1), ("tests/dt-maze", 2)])
class TestSchedulersOnDecisionTrees:
    """Can a decision tree of this depth make the choices of a scheduler?

    The choices that a tree itself makes can be made, by that tree or another, and for a scheduler changed in some states the verdict is the engine's.
    """

    @staticmethod
    def load(project, depth):
        factory, _task = load_sketch(project)
        return factory.reset_tree(depth, general=True)

    def test_the_choices_that_a_tree_makes_can_be_made(self, project, depth):
        colored_mdp = self.load(project, depth)
        parameter_space, coloring = colored_mdp.parameter_space, colored_mdp.coloring
        for _ in range(3):
            chosen = list(coloring.selectCompatibleChoices(parameter_space.pick_random().native))
            values = take_together(coloring, parameter_space, chosen)
            assert values is not None
            found = parameter_space.construct_assignment(values)
            assert set(coloring.selectCompatibleChoices(found.native)) == set(chosen)

    def test_the_verdict_on_a_changed_scheduler_is_the_engines(self, project, depth):
        colored_mdp = self.load(project, depth)
        parameter_space, coloring = colored_mdp.parameter_space, colored_mdp.coloring
        row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
        rng = random.Random(depth)
        verdicts = set()
        for _ in range(3):
            chosen = set(coloring.selectCompatibleChoices(parameter_space.pick_random().native))
            for changes in range(4):  # the scheduler of a tree, then with a state or two or three playing another choice
                states = [bisect.bisect_right(row_groups, choice) - 1 for choice in sorted(chosen)]
                state = rng.choice([state for state in states if row_groups[state + 1] - row_groups[state] > 1])
                if changes:
                    chosen = {choice for choice in chosen if not row_groups[state] <= choice < row_groups[state + 1]}
                    chosen.add(rng.randrange(row_groups[state], row_groups[state + 1]))
                consistent, _selection = colored_mdp.are_choices_consistent(choice_mask(row_groups[-1], chosen), parameter_space)
                assert (take_together(coloring, parameter_space, sorted(chosen)) is not None) == consistent, sorted(chosen)
                verdicts.add(consistent)
        assert verdicts == {True, False}


class TestDataIsSubstituted:
    def test_a_state_column_and_a_choice_column(self):
        """Three states with two choices: the cell of a state and the action of a choice are data, a threshold on the cell and the action are parameters."""
        parameter_space = parameter_space_of([[0, 1, 2, 3], [0, 1]], ["threshold", "action"])
        builder = ColoringBuilder([0, 2, 4, 6], 2)
        threshold, action = builder.parameters
        cell, played = builder.state_column([0, 1, 2]), builder.choice_column([0, 1, 0, 1, 0, 1])
        builder.color(range(6), builder.template(z3.And(threshold > cell, action == played)))
        coloring = builder.build()
        threshold, action = z3.Int("threshold"), z3.Int("action")
        assert equivalent(choice_formula(coloring, 0, parameter_space), z3.And(threshold > 0, action == 0))
        assert equivalent(choice_formula(coloring, 3, parameter_space), z3.And(threshold > 1, action == 1))
        assert equivalent(choice_formula(coloring, 4, parameter_space), z3.And(threshold > 2, action == 0))

    def test_arithmetic_and_conditions_on_data_are_computed(self):
        parameter_space = parameter_space_of([[0, 1, 2, 3, 4, 5, 6]], ["level"])
        builder = ColoringBuilder([0, 1, 2], 1)
        (p,) = builder.parameters
        x = builder.state_column([2, 5])
        builder.color(0, builder.template(z3.If(x * 2 - 3 > 0, p == x + 1, p == 0)))
        builder.color(1, builder.template(z3.If(x * 2 - 3 > 7, p == x + 1, p == 0)))
        coloring = builder.build()
        level = z3.Int("level")
        assert equivalent(choice_formula(coloring, 0, parameter_space), level == 3)  # 2 * 2 - 3 = 1 > 0
        assert equivalent(choice_formula(coloring, 1, parameter_space), level == 0)  # 2 * 5 - 3 = 7, not above 7

    def test_in_bits_becomes_the_options_of_the_set_bits_of_the_word(self):
        parameter_space = parameter_space_of([[0, 1]], ["mode"])
        builder = ColoringBuilder([0, 1, 2, 3, 4], 1)
        (p,) = builder.parameters
        mask = builder.state_column([0b0101, 0, -(2**63), -1])
        builder.color(range(4), builder.template(ColoringBuilder.in_bits(p, mask)))
        coloring = builder.build()
        mode = z3.Int("mode")
        assert equivalent(choice_formula(coloring, 0, parameter_space), z3.Or(mode == 0, mode == 2))
        assert z3.is_false(choice_formula(coloring, 1, parameter_space))
        assert equivalent(choice_formula(coloring, 2, parameter_space), mode == 63), "the sign bit of the word is the option 63"
        assert equivalent(choice_formula(coloring, 3, parameter_space), z3.And(mode >= 0, mode <= 63))

    def test_in_set_becomes_the_options_of_the_set(self):
        parameter_space = parameter_space_of([[0, 1, 2, 3]], ["mode"])
        builder = ColoringBuilder([0, 2], 1)
        (p,) = builder.parameters
        builder.color(0, builder.template(z3.Or(p == 1, p == 3)))
        coloring = builder.build()
        mode = z3.Int("mode")
        assert equivalent(choice_formula(coloring, 0, parameter_space), z3.Or(mode == 1, mode == 3))


class TestUncoloredAndIrrelevantStates:
    """Four states: 0 and 2 are relevant, 1 is irrelevant and has no choice at all, 3 is irrelevant with three choices that are colored all the same, which the
    engine takes no notice of, as it enables only the first choice of an irrelevant state."""

    ROW_GROUPS = [0, 2, 2, 4, 7]

    def build(self):
        builder = ColoringBuilder(self.ROW_GROUPS, 1, state_is_relevant=self.relevance())
        (p,) = builder.parameters
        x = builder.state_column([0, 7, 1, 9])
        equal, different = builder.template(p == x), builder.template(p != x)
        builder.color([0, 2], equal)
        builder.color([1, 3], different)
        builder.color([4, 5, 6], equal)
        return parameter_space_of([[0, 1, 2]], ["level"]), builder.build()

    @staticmethod
    def relevance():
        relevant = stormpy.datastructures.BitVector(4)
        relevant.set(0, True)
        relevant.set(2, True)
        return relevant

    def test_the_formula_agrees_with_the_engine_at_every_choice(self):
        parameter_space, coloring = self.build()
        for value in range(3):
            selected = enabled(coloring, parameter_space, (value,))
            for choice in range(7):
                assert holds(choice_formula(coloring, choice, parameter_space), (value,), parameter_space) == bool(selected[choice]), (choice, value)

    def test_only_the_first_choice_of_an_irrelevant_state_is_enabled(self):
        parameter_space, coloring = self.build()
        assert [z3.is_true(choice_formula(coloring, choice, parameter_space)) for choice in (4, 5, 6)] == [True, False, False]
        assert [z3.is_false(choice_formula(coloring, choice, parameter_space)) for choice in (4, 5, 6)] == [False, True, True]

    def test_a_choice_belongs_to_the_state_that_has_it_not_to_an_empty_state_before(self):
        parameter_space, coloring = self.build()
        level = z3.Int("level")
        assert equivalent(choice_formula(coloring, 2, parameter_space), level == 1), "choice 2 is in state 2, whose column holds 1 (state 1, with no choice, 7)"
        assert equivalent(choice_formula(coloring, 3, parameter_space), level != 1)

    def test_an_uncolored_choice_is_enabled_by_every_assignment(self):
        builder = ColoringBuilder([0, 2], 1)
        (p,) = builder.parameters
        builder.color(0, builder.template(p == 1))
        assert z3.is_true(choice_formula(builder.build(), 1, parameter_space_of([[0, 1]], ["level"])))


class TestLongChains:
    def test_more_disjuncts_than_the_limit_on_recursion_are_no_problem(self):
        """A decision tree of depth 10 has a thousand leaves, and its color is a disjunction with a disjunct for each, which the builder makes a chain of OrF
        nodes as long as that: a recursive walk over the node table would fail on it."""
        domains = [[0, 1, 2], [0, 1, 2, 3, 4]]
        parameter_space = parameter_space_of(domains, ["row", "column"])
        builder = ColoringBuilder([0, 2], 2)
        p0, p1 = builder.parameters
        disjuncts = [z3.And(p0 == i % 3, p1 == (i // 3) % 5, p0 + p1 > i % 7 - 3) for i in range(sys.getrecursionlimit() + 200)]
        builder.color(0, builder.template(z3.Or(*disjuncts)))
        coloring = builder.build()
        formula = choice_formula(coloring, 0, parameter_space)
        for values in itertools.product(*domains):
            assert holds(formula, values, parameter_space) == bool(enabled(coloring, parameter_space, values)[0]), values


class TestLiftedStandardColorings:
    @pytest.mark.parametrize("project", ["tests/smpmc-tiny", "tests/generic-maze"])
    def test_the_formula_is_the_conjunction_of_the_pairs(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        parameter_space = colored_mdp.parameter_space
        general = general_colored_mdp(colored_mdp).coloring
        names = [parameter_space.parameter_name(parameter) for parameter in range(parameter_space.num_parameters)]
        for choice, pairs in enumerate(colored_mdp.coloring.getChoiceToAssignment()):
            expected = z3.And([z3.Int(names[parameter]) == option for parameter, option in pairs]) if pairs else z3.BoolVal(True)
            assert equivalent(choice_formula(general, choice, parameter_space), expected), choice

    def test_the_formula_is_over_the_holes_of_the_sketch(self):
        """The holes of smpmc-tiny are x1, x2 and x3, and the coloring of its choice 0 is made of the pairs (x1, option 0) and (x3, option 0)."""
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        formula = choice_formula(general_colored_mdp(colored_mdp).coloring, 0, colored_mdp.parameter_space)
        assert names_in(formula) == {"x1", "x3"}
        assert equivalent(formula, z3.And(z3.Int("x1") == 0, z3.Int("x3") == 0))


@pytest.mark.parametrize(("project", "depth"), [("tests/dt-maze", 0), ("tests/dt-maze", 2), ("tests/dt-orchard", 1)])
class TestDecisionTrees:
    def test_the_formula_of_a_choice_holds_exactly_where_the_choice_is_enabled(self, project, depth):
        """Only the choices of the states that the tree reaches are enabled or not (the coloring explores the states from the initial one), so the others are
        left out."""
        factory, _task = load_sketch(project)
        colored_mdp = factory.reset_tree(depth, general=True)
        parameter_space, coloring = colored_mdp.parameter_space, colored_mdp.coloring
        names = {parameter_space.parameter_name(parameter) for parameter in range(parameter_space.num_parameters)}
        row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
        rng = random.Random(depth)
        checked, used = 0, set()
        for _ in range(6):
            assignment = parameter_space.pick_random()
            values = [assignment.parameter_options(parameter)[0] for parameter in range(parameter_space.num_parameters)]
            selected = coloring.selectCompatibleChoices(assignment.native)
            reached = {bisect.bisect_right(row_groups, choice) - 1 for choice in range(row_groups[-1]) if selected[choice]}
            for choice in rng.sample(range(row_groups[-1]), min(30, row_groups[-1])):
                if bisect.bisect_right(row_groups, choice) - 1 in reached:
                    checked += 1
                    formula = choice_formula(coloring, choice, parameter_space)
                    if checked <= 3:  # the color of a choice has a disjunct for every leaf, so each formula has the names of the tree, and z3util is slow
                        used |= names_in(formula)
                    assert holds(formula, values, parameter_space) == bool(selected[choice]), (choice, values)
        assert checked > 0
        assert used and used <= names


class TestTheExampleOfTheDocstring:
    def test_the_color_of_a_choice_goes_into_a_solver_of_your_own(self):
        """`fast` in the cells below a threshold, `slow` from it on, over three cells (a state each, with a fast and a slow choice)."""
        parameter_space = parameter_space_of([[0, 1, 2, 3]], ["threshold"])
        builder = ColoringBuilder([0, 2, 4, 6], 1)
        (threshold,) = builder.parameters
        cell = builder.state_column([0, 1, 2])
        fast, slow = builder.template(threshold > cell), builder.template(threshold <= cell)
        for state in range(3):
            builder.color(2 * state, fast)
            builder.color(2 * state + 1, slow)
        coloring = builder.build(parameter_space)
        assert paynt.utils.coloring.is_general_coloring(coloring)

        solver = z3.Solver()
        for parameter, variable in enumerate(parameter_variables(parameter_space)):
            solver.add(z3.Or([variable == option for option in parameter_space.parameter_options(parameter)]))
        solver.add(choice_formula(coloring, 2, parameter_space))  # fast in cell 1: the threshold is above 1
        solver.add(choice_formula(coloring, 5, parameter_space))  # slow in cell 2: the threshold is at most 2
        assert solver.check() == z3.sat
        assert solver.model()[z3.Int("threshold")].as_long() == 2
        solver.add(choice_formula(coloring, 1, parameter_space))  # slow in cell 0: the threshold is 0, which is not above 1
        assert solver.check() == z3.unsat


class TestErrors:
    @pytest.mark.parametrize("with_parameter_space", [False, True])
    def test_a_standard_coloring_is_refused_with_a_hint(self, with_parameter_space):
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        with pytest.raises(TypeError, match=r"not from a Coloring: .* the conjunction of its \(parameter, option\) pairs"):
            choice_formula(colored_mdp.coloring, 0, *([colored_mdp.parameter_space] if with_parameter_space else []))

    @pytest.mark.parametrize("with_parameter_space", [False, True])
    def test_a_choice_that_does_not_exist(self, with_parameter_space):
        coloring = ColoringBuilder([0, 2], 1).build()
        extra = [parameter_space_of([[0, 1]], ["level"])] if with_parameter_space else []
        assert z3.is_true(choice_formula(coloring, 1, *extra))
        with pytest.raises(IndexError, match="choice 2 is out of range: the coloring has 2 choices"):
            choice_formula(coloring, 2, *extra)

    def test_a_parameter_space_with_fewer_parameters_than_the_color_depends_on(self):
        builder = ColoringBuilder([0, 1], 3)
        builder.color(0, builder.template(builder.parameters[2] == 1))
        coloring = builder.build()
        with pytest.raises(ValueError, match="choice 0 depends on parameter 2, but the parameter space has only 2 parameters"):
            choice_formula(coloring, 0, parameter_space_of([[0, 1], [0, 1]], ["a", "b"]))
