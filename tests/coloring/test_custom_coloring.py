"""Colorings written from scratch with ColoringBuilder, end to end.

The model is a corridor of N = 6 cells. In every cell the agent either walks the slow way (2 time units, 90 % one cell forward, else it stays) or the fast way
(1 time unit, 50 % two cells forward, 20 % it stays, 30 % back to cell 0), and the property is the expected time to the end. A *policy class* is a coloring:
which parameter assignment plays which action in which cell. Two are tried:

- one parameter per cell, taking `slow` or `fast` -- every color is a conjunction of (parameter == option) pairs, so the builder gives the standard coloring and
  every engine works;
- one threshold T, `fast` in the cells below T and `slow` from T on -- the colors `T > cell` and `T <= cell` are not conjunctions, so it is a general coloring,
  which OneByOne, CEGIS and SMPMC accept.

The expected times are computed here from the transition probabilities, with numpy, so the expected optimum does not depend on Storm or PAYNT.

The module also checks the builder's formula language against Python on random formulas, and the example of its docstring.
"""

from __future__ import annotations

import itertools
import json
import operator
import random
import textwrap

import numpy as np
import pytest
import z3

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.parser.sketch
import paynt.synthesizer.synthesizer
import paynt.utils.coloring_builder

from helpers.helper import load_colored_mdp

N = 6
CORRIDOR = textwrap.dedent("""
    mdp

    const int N = 6;

    module corridor
        x : [0..N] init 0;
        [slow] x < N -> 0.9 : (x' = x + 1) + 0.1 : (x' = x);
        [fast] x < N -> 0.5 : (x' = min(x + 2, N)) + 0.2 : (x' = x) + 0.3 : (x' = 0);
        [done] x = N -> (x' = N);
    endmodule

    rewards "time"
        [slow] true : 2;
        [fast] true : 1;
    endrewards

    label "goal" = x = N;
    """)
ACTIONS = ["slow", "fast"]


def expected_time(policy):
    """The expected time from cell 0 to the end when cell x plays policy[x] (0 = slow, 1 = fast): the linear system T(x) = cost + sum P(x, y) T(y)."""
    matrix, cost = np.eye(N), np.zeros(N)
    for cell, action in enumerate(policy):
        if action == 0:
            cost[cell], outcomes = 2, [(0.9, cell + 1), (0.1, cell)]
        else:
            cost[cell], outcomes = 1, [(0.5, min(cell + 2, N)), (0.2, cell), (0.3, 0)]
        for probability, target in outcomes:
            if target < N:
                matrix[cell, target] -= probability
    return float(np.linalg.solve(matrix, cost)[0])


def threshold_policy(threshold):
    return [1 if cell < threshold else 0 for cell in range(N)]


BEST_PER_CELL = min(expected_time(policy) for policy in itertools.product([0, 1], repeat=N))
BEST_THRESHOLD = min(expected_time(threshold_policy(threshold)) for threshold in range(N + 1))


def corridor(tmp_path, props='R{"time"}min=? [F "goal"]'):
    """(factory, task) of the corridor; the factory is that of a decision tree, but its quotient MDP is just the corridor once the don't-care action is off."""
    (tmp_path / "sketch.templ").write_text(CORRIDOR)
    (tmp_path / "sketch.props").write_text(props + "\n")
    return paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"), task_kwargs={"add_dont_care_action": False})


def cells_and_actions(mdp):
    """The cell of every state, and the action (0 = slow, 1 = fast, None = the finish) of every choice."""
    cells = [json.loads(str(mdp.state_valuations.get_json(state)))["x"] for state in range(mdp.nr_states)]
    labels = [next(iter(mdp.choice_labeling.get_labels_of_choice(choice))) for choice in range(mdp.nr_choices)]
    return cells, [ACTIONS.index(label) if label in ACTIONS else None for label in labels]


def per_cell_colored_mdp(tmp_path, props='R{"time"}min=? [F "goal"]'):
    """One parameter per cell, `slow` or `fast`: the standard coloring."""
    factory, task = corridor(tmp_path, props)
    mdp = factory.underlying_mdp
    cells, actions = cells_and_actions(mdp)
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for cell in range(N):
        parameter_space.add_parameter(f"cell{cell}", ACTIONS)
    builder = paynt.utils.coloring_builder.ColoringBuilder(mdp.nondeterministic_choice_indices, N)
    for state in range(mdp.nr_states):
        for choice in range(mdp.nondeterministic_choice_indices[state], mdp.nondeterministic_choice_indices[state + 1]):
            if actions[choice] is not None:
                builder.color(choice, builder.template(builder.parameters[cells[state]] == actions[choice]))
    coloring = builder.build(parameter_space)
    return paynt.colored_mdp.ColoredMdp(mdp, parameter_space, coloring), task, coloring


def threshold_colored_mdp(tmp_path, props='R{"time"}min=? [F "goal"]'):
    """One threshold: `fast` in the cells below it, `slow` from it on: a general coloring."""
    factory, task = corridor(tmp_path, props)
    mdp = factory.underlying_mdp
    cells, actions = cells_and_actions(mdp)
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    parameter_space.add_parameter("threshold", list(range(N + 1)))
    builder = paynt.utils.coloring_builder.ColoringBuilder(mdp.nondeterministic_choice_indices, 1)
    threshold = builder.parameters[0]
    cell = builder.state_column(cells)
    fast, slow = builder.template(threshold > cell), builder.template(threshold <= cell)
    for choice, action in enumerate(actions):
        if action is not None:
            builder.color(choice, fast if action == 1 else slow)
    coloring = builder.build(parameter_space)
    return paynt.colored_mdp.ColoredMdp(mdp, parameter_space, coloring), task, coloring


class TestTheColoringsAreWhatTheyShouldBe:
    def test_one_parameter_per_cell_is_the_standard_coloring(self, tmp_path):
        colored_mdp, _task, coloring = per_cell_colored_mdp(tmp_path)
        assert type(coloring).__name__ == "Coloring"
        assert not colored_mdp.has_general_coloring
        assert colored_mdp.parameter_space.size == 2**N

    def test_a_threshold_is_a_general_coloring(self, tmp_path):
        colored_mdp, _task, coloring = threshold_colored_mdp(tmp_path)
        assert type(coloring).__name__ == "ColoringGeneral"
        assert colored_mdp.has_general_coloring
        assert colored_mdp.parameter_space.size == N + 1

    def test_the_oracle_agrees_with_model_checking_on_every_policy(self, tmp_path):
        """The expected times below are not trusted blindly: Storm gives the same on the DTMC of every assignment of both policy classes."""
        for builder in (per_cell_colored_mdp, threshold_colored_mdp):
            colored_mdp, task, _coloring = builder(tmp_path)
            space = colored_mdp.parameter_space
            for combination in space.all_combinations():
                model = colored_mdp.build_assignment(space.construct_assignment(combination))
                policy = list(combination) if builder is per_cell_colored_mdp else threshold_policy(combination[0])
                assert model.model_check_property(task.get_property()).value == pytest.approx(expected_time(policy), rel=1e-6)

    def test_every_assignment_enables_exactly_one_choice_in_every_state(self, tmp_path):
        """Definition 2: the finish has a single choice, left uncolored, and the others are exclusive."""
        for builder in (per_cell_colored_mdp, threshold_colored_mdp):
            colored_mdp, _task, coloring = builder(tmp_path)
            space = colored_mdp.parameter_space
            row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
            for combination in space.all_combinations():
                selected = coloring.selectCompatibleChoices(space.construct_assignment(combination).native)
                for state in range(len(row_groups) - 1):
                    assert sum(selected[choice] for choice in range(row_groups[state], row_groups[state + 1])) == 1

    def test_the_threshold_plays_fast_below_it_and_slow_from_it_on(self, tmp_path):
        colored_mdp, _task, _coloring = threshold_colored_mdp(tmp_path)
        mdp = colored_mdp.underlying_mdp
        cells, actions = cells_and_actions(mdp)
        for threshold in range(N + 1):
            assignment = colored_mdp.parameter_space.construct_assignment((threshold,))
            _model, selected = colored_mdp.build(assignment)
            played = {
                cells[state]: actions[choice]
                for state in range(mdp.nr_states)
                for choice in range(mdp.nondeterministic_choice_indices[state], mdp.nondeterministic_choice_indices[state + 1])
                if selected[choice] and actions[choice] is not None
            }
            assert played == dict(enumerate(threshold_policy(threshold)))


@pytest.mark.parametrize("method", ["onebyone", "ar", "cegis", "hybrid", "smpmc"])
class TestStandardColoringEngines:
    def test_the_optimum_is_the_best_of_the_policies(self, tmp_path, method):
        colored_mdp, task, _coloring = per_cell_colored_mdp(tmp_path)
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        assert result.success
        assert result.value == pytest.approx(BEST_PER_CELL, rel=1e-6)

    def test_the_policy_found_is_as_good_as_it_says(self, tmp_path, method):
        colored_mdp, task, _coloring = per_cell_colored_mdp(tmp_path)
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        policy = [result.assignment.parameter_options(cell)[0] for cell in range(N)]
        assert expected_time(policy) == pytest.approx(result.value, rel=1e-6)

    def test_the_selected_choices_are_the_policy_found(self, tmp_path, method):
        colored_mdp, task, _coloring = per_cell_colored_mdp(tmp_path)
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        policy = [result.assignment.parameter_options(cell)[0] for cell in range(N)]
        cells, actions = cells_and_actions(colored_mdp.underlying_mdp)
        row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
        reached = []
        for state, cell in enumerate(cells):
            enabled = [choice for choice in range(row_groups[state], row_groups[state + 1]) if result.selected_choices[choice]]
            assert len(enabled) <= 1
            if enabled:
                reached.append(cell)
                if cell < N:
                    assert actions[enabled[0]] == policy[cell]
        assert 0 in reached and N in reached


@pytest.mark.parametrize("method", ["onebyone", "cegis", "smpmc"])
class TestGeneralColoringEngines:
    def test_the_optimum_is_the_best_threshold(self, tmp_path, method):
        colored_mdp, task, _coloring = threshold_colored_mdp(tmp_path)
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        assert result.success
        assert result.value == pytest.approx(BEST_THRESHOLD, rel=1e-6)
        threshold = result.assignment.parameter_options(0)[0]
        assert expected_time(threshold_policy(threshold)) == pytest.approx(result.value, rel=1e-6)

    @pytest.mark.parametrize(("bound", "satisfiable"), [(9.7, True), (9.6, False)])
    def test_the_verdict_on_a_threshold_on_the_time(self, tmp_path, method, bound, satisfiable):
        """The best threshold takes 9.644...: 9.7 can be met, 9.6 not."""
        colored_mdp, task, _coloring = threshold_colored_mdp(tmp_path, f'R{{"time"}}<={bound} [F "goal"]')
        assert paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run().success is satisfiable

    def test_the_selected_choices_are_the_policy_found(self, tmp_path, method):
        """One choice for every state the policy reaches -- not all of them: the fast way skips cells -- and each is the action of the threshold."""
        colored_mdp, task, _coloring = threshold_colored_mdp(tmp_path)
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        threshold = result.assignment.parameter_options(0)[0]
        cells, actions = cells_and_actions(colored_mdp.underlying_mdp)
        row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
        reached = []
        for state, cell in enumerate(cells):
            enabled = [choice for choice in range(row_groups[state], row_groups[state + 1]) if result.selected_choices[choice]]
            assert len(enabled) <= 1
            if enabled:
                reached.append(cell)
                if cell < N:
                    assert actions[enabled[0]] == threshold_policy(threshold)[cell]
        assert 0 in reached and N in reached


@pytest.mark.parametrize("method", ["ar", "hybrid"])
def test_ar_and_hybrid_refuse_the_general_coloring_with_a_hint(tmp_path, method):
    colored_mdp, task, _coloring = threshold_colored_mdp(tmp_path)
    with pytest.raises(NotImplementedError, match="general coloring for method .*use --method onebyone, cegis or smpmc"):
        paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method)


class TestDocstringExample:
    """The example of paynt.utils.coloring_builder's docstring, on a real model: a flag column, a color that is not a conjunction, one that is."""

    def test_the_example_builds_a_general_coloring_that_does_what_it_says(self):
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        parameter_space.add_parameter("environment", [0, 1, 2])
        parameter_space.add_parameter("action", ["left", "straight", "right"])

        mdp = colored_mdp.underlying_mdp
        builder = paynt.utils.coloring_builder.ColoringBuilder(mdp.nondeterministic_choice_indices, parameter_space.num_parameters)
        p, action = builder.parameters[0], builder.parameters[1]
        flags = [state % 2 for state in range(mdp.nr_states)]
        flag = builder.state_column(flags)
        builder.color([3, 7], builder.template(z3.Or(z3.Or(p == 0, p == 2), flag == 1)))
        builder.color(5, builder.template(action == 2))
        coloring = builder.build(parameter_space)
        assert type(coloring).__name__ == "ColoringGeneral"
        example = paynt.colored_mdp.ColoredMdp(mdp, parameter_space, coloring)
        assert example.has_general_coloring

        row_groups = mdp.nondeterministic_choice_indices
        state_of = {choice: state for state in range(mdp.nr_states) for choice in range(row_groups[state], row_groups[state + 1])}
        for environment, shared_action in itertools.product(range(3), range(3)):
            assignment = parameter_space.construct_assignment((environment, shared_action))
            selected = coloring.selectCompatibleChoices(assignment.native)
            for choice in range(mdp.nr_choices):
                if choice in (3, 7):
                    expected = environment in (0, 2) or flags[state_of[choice]] == 1
                elif choice == 5:
                    expected = shared_action == 2
                else:
                    expected = True  # uncolored: always enabled
                assert selected[choice] == expected, (choice, environment, shared_action)

    def test_the_same_colors_without_the_flag_are_the_standard_coloring(self):
        """With every color a conjunction of (parameter == option) pairs, build() returns the standard coloring, as the docstring says."""
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        parameter_space.add_parameter("environment", [0, 1, 2])
        parameter_space.add_parameter("action", ["left", "straight", "right"])
        builder = paynt.utils.coloring_builder.ColoringBuilder(colored_mdp.underlying_mdp.nondeterministic_choice_indices, 2)
        p, action = builder.parameters
        builder.color([3, 7], builder.template(z3.And(p == 0, action == 2)))
        builder.color(5, builder.template(action == 2))
        assert type(builder.build(parameter_space)).__name__ == "Coloring"


class FormulaGenerator:
    """Random formulas over three parameters (value = option index), one data column and a bit mask, built together with the Python function that evaluates
    them: the reference the builder's formula language is held to."""

    COMPARISONS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]

    def __init__(self, rng, parameters, column, mask):
        self.rng = rng
        self.parameters = parameters
        self.column = column
        self.mask = mask

    def term(self, depth):
        rng = self.rng
        kind = rng.choice(["parameter", "literal", "column"] if depth == 0 else ["parameter", "literal", "column", "add", "sub", "mul", "neg", "if"])
        if kind == "parameter":
            index = rng.randrange(len(self.parameters))
            return self.parameters[index], lambda values, data, index=index: values[index]
        if kind == "literal":
            literal = rng.randint(-1, 3)
            return z3.IntVal(literal), lambda values, data, literal=literal: literal
        if kind == "column":
            return self.column, lambda values, data: data["column"]
        if kind == "neg":
            expression, function = self.term(depth - 1)
            return -expression, lambda values, data: -function(values, data)
        if kind == "if":
            condition, test = self.boolean(depth - 1)
            (then_expression, then_function), (else_expression, else_function) = self.term(depth - 1), self.term(depth - 1)
            return (
                z3.If(condition, then_expression, else_expression),
                lambda values, data: then_function(values, data) if test(values, data) else else_function(values, data),
            )
        (left, left_function), (right, right_function) = self.term(depth - 1), self.term(depth - 1)
        combine = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}[kind]
        return combine(left, right), lambda values, data: combine(left_function(values, data), right_function(values, data))

    def boolean(self, depth):
        rng = self.rng
        kind = rng.choice(["compare", "options", "bits"] if depth == 0 else ["compare", "options", "bits", "not", "and", "or", "implies", "if"])
        if kind == "compare":
            compare = rng.choice(self.COMPARISONS)
            (left, left_function), (right, right_function) = self.term(max(depth - 1, 0)), self.term(max(depth - 1, 0))
            return compare(left, right), lambda values, data: compare(left_function(values, data), right_function(values, data))
        if kind == "options":
            index = rng.randrange(len(self.parameters))
            options = rng.sample([0, 1, 2], rng.randint(1, 3))
            return z3.Or([self.parameters[index] == option for option in options]), lambda values, data: values[index] in options
        if kind == "bits":
            index = rng.randrange(len(self.parameters))
            return paynt.utils.coloring_builder.ColoringBuilder.in_bits(self.parameters[index], self.mask), lambda values, data: bool(
                (data["mask"] >> values[index]) & 1
            )
        if kind == "not":
            expression, function = self.boolean(depth - 1)
            return z3.Not(expression), lambda values, data: not function(values, data)
        if kind == "if":
            (condition, test), (then_expression, then_function), (else_expression, else_function) = (
                self.boolean(depth - 1),
                self.boolean(depth - 1),
                self.boolean(depth - 1),
            )
            return (
                z3.If(condition, then_expression, else_expression),
                lambda values, data: then_function(values, data) if test(values, data) else else_function(values, data),
            )
        operands = [self.boolean(depth - 1) for _ in range(rng.randint(2, 3))]
        expressions = [expression for expression, _ in operands]
        functions = [function for _, function in operands]
        if kind == "and":
            return z3.And(expressions), lambda values, data: all(function(values, data) for function in functions)
        if kind == "or":
            return z3.Or(expressions), lambda values, data: any(function(values, data) for function in functions)
        return z3.Implies(expressions[0], expressions[1]), lambda values, data: (not functions[0](values, data)) or functions[1](values, data)


@pytest.mark.parametrize("seed", range(6))
class TestRandomFormulasAgainstPython:
    """ColoringGeneral decides every random formula exactly on a full assignment, and on a family it never leaves out a choice that some member enables (the
    selection over-approximates, as AR and SMPMC rely on)."""

    NUM_FORMULAS = 40
    DOMAINS = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]

    def build(self, seed):
        rng = random.Random(seed)
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        for index, domain in enumerate(self.DOMAINS):
            parameter_space.add_parameter(f"p{index}", domain)
        # one state: NUM_FORMULAS colored choices and an uncolored one, so that the state keeps a choice whatever the formulas say
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, self.NUM_FORMULAS + 1], len(self.DOMAINS))
        data = {"column": rng.randint(-1, 3), "mask": rng.randint(0, 2**8 - 1)}
        generator = FormulaGenerator(rng, builder.parameters, builder.state_column([data["column"]]), builder.state_column([data["mask"]]))
        functions = []
        for choice in range(self.NUM_FORMULAS):
            expression, function = generator.boolean(rng.randint(1, 3))
            builder.color(choice, builder.template(expression))
            functions.append(function)
        return parameter_space, builder.build(), functions, data, rng

    def test_a_full_assignment_enables_exactly_the_choices_whose_formula_holds(self, seed):
        parameter_space, coloring, functions, data, _rng = self.build(seed)
        for values in itertools.product(*self.DOMAINS):
            selected = coloring.selectCompatibleChoices(parameter_space.construct_assignment(values).native)
            for choice, function in enumerate(functions):
                assert selected[choice] == function(values, data), (choice, values)
            assert selected[self.NUM_FORMULAS]

    def test_a_family_enables_every_choice_that_some_member_enables(self, seed):
        parameter_space, coloring, functions, data, rng = self.build(seed)
        for _ in range(30):
            family = parameter_space.copy()
            options = []
            for parameter in range(parameter_space.num_parameters):
                chosen = sorted(rng.sample(self.DOMAINS[parameter], rng.randint(1, 3)))
                family.parameter_set_options(parameter, chosen)
                options.append(chosen)
            selected = coloring.selectCompatibleChoices(family.native)
            for choice, function in enumerate(functions):
                if any(function(values, data) for values in itertools.product(*options)):
                    assert selected[choice], (choice, options)

    def test_a_family_that_cannot_enable_a_choice_often_leaves_it_out(self, seed):
        """Soundness alone is satisfied by enabling everything: where the formula is decided by the family (all members agree it fails), the selection is
        usually tight, so a sizeable share of the impossible choices is still left out."""
        parameter_space, coloring, functions, data, rng = self.build(seed)
        impossible = left_out = 0
        for _ in range(30):
            family = parameter_space.copy()
            options = []
            for parameter in range(parameter_space.num_parameters):
                chosen = sorted(rng.sample(self.DOMAINS[parameter], rng.randint(1, 2)))
                family.parameter_set_options(parameter, chosen)
                options.append(chosen)
            selected = coloring.selectCompatibleChoices(family.native)
            for choice, function in enumerate(functions):
                if not any(function(values, data) for values in itertools.product(*options)):
                    impossible += 1
                    left_out += not selected[choice]
        assert impossible > 0
        assert left_out >= impossible / 2
