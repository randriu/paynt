"""ColoringGeneral.relevantParameters: the primitive behind SMPMC's conflict minimization (Theorem 6) on general colorings.

Its contract: widening the domain of any parameter outside the returned set leaves the selected choices at the given states unchanged. That is exactly why a
refutation of the sub-MDP those states induce carries over to every family that agrees with the refuted one on the returned parameters. It is tested on that
statement directly -- on decision trees, where it matters (every state's static supports are all the tree's parameters), and on lifted standard colorings -- and
on how tight it is, so that it cannot pass by returning everything.
"""

from __future__ import annotations

import random

import pytest
import stormpy.storage
import z3

import paynt.parameter_space.parameter_space
import paynt.parser.sketch
import paynt.utils.coloring_builder

from helpers.helper import general_colored_mdp, get_sketch_paths, load_colored_mdp

TREES = [("tests/dt-orchard", 1), ("tests/dt-orchard", 2), ("tests/dt-maze", 1), ("tests/dt-maze", 2)]
LIFTED = ["tests/smpmc-tiny", "tests/mdp-family-avoid-8-2-easy"]


def tree_colored_mdp(project, depth):
    sketch_path, props_path = get_sketch_paths(project)
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory.reset_tree(depth, general=True)


def random_family(rng, parameter_space, fixed_probability):
    """A random subfamily of parameter_space in which each parameter is fixed to a random option with the given probability.

    Returns (family, fixed).
    """
    family = parameter_space.copy()
    fixed = set()
    for parameter in range(parameter_space.num_parameters):
        if rng.random() < fixed_probability:
            family.parameter_set_options(parameter, [rng.choice(parameter_space.parameter_options(parameter))])
            fixed.add(parameter)
    return family, fixed


def widen_outside(rng, family, keep, parameter_space):
    """A random superset of family that agrees with it on every parameter in keep."""
    wide = family.copy()
    for parameter in range(parameter_space.num_parameters):
        if parameter in keep:
            continue
        extra = [option for option in parameter_space.parameter_options(parameter) if rng.random() < 0.5]
        wide.parameter_set_options(parameter, sorted(set(family.parameter_options(parameter)) | set(extra)))
    return wide


def reachable(colored_mdp, family):
    """(selected choices, reachable states, every choice of those states) of the sub-MDP the family induces."""
    sub_mdp, selected = colored_mdp.build(family)
    states = list(sub_mdp.underlying_mdp_state_map)
    row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
    return selected, states, [choice for state in states for choice in range(row_groups[state], row_groups[state + 1])]


def selection_changes(colored_mdp, selected, choices, wide):
    """Whether the choices' selection under the wider family differs from selected."""
    selection = colored_mdp.coloring.selectCompatibleChoices(wide.native)
    return any(selection[choice] != selected[choice] for choice in choices)


def check_widening_keeps_the_selection(colored_mdp, seed, families, widenings, fixed_probability):
    """The contract, on random families.

    Returns how many families had a fixed parameter outside the relevant set (so the check was not vacuous).
    """
    parameter_space = colored_mdp.parameter_space
    rng = random.Random(seed)
    dropped_some = 0
    for _ in range(families):
        family, fixed = random_family(rng, parameter_space, fixed_probability)
        selected, states, choices = reachable(colored_mdp, family)
        relevant = set(colored_mdp.coloring.relevantParameters(family.native, states))
        dropped_some += bool(fixed - relevant)
        for _ in range(widenings):
            wide = widen_outside(rng, family, relevant, parameter_space)
            assert not selection_changes(colored_mdp, selected, choices, wide), f"widening outside {sorted(relevant)} changed the selection"
    return dropped_some


@pytest.mark.parametrize(("project", "depth"), TREES)
class TestOnDecisionTrees:
    def test_widening_outside_the_relevant_parameters_keeps_the_selection(self, project, depth):
        colored_mdp = tree_colored_mdp(project, depth)
        dropped_some = check_widening_keeps_the_selection(colored_mdp, seed=depth, families=25, widenings=4, fixed_probability=0.7)
        assert dropped_some > 0, "every fixed parameter was relevant every time: the check proves nothing about dropping"

    def test_each_relevant_parameter_matters(self, project, depth):
        """Widening a single fixed member of the relevant set, alone, usually changes the selection: the set is not padded, and a test like the one above would
        notice if it were too small."""
        colored_mdp = tree_colored_mdp(project, depth)
        parameter_space = colored_mdp.parameter_space
        rng = random.Random(100 + depth)
        members = changed = 0
        for _ in range(20):
            family, fixed = random_family(rng, parameter_space, 0.7)
            selected, states, choices = reachable(colored_mdp, family)
            for parameter in sorted(fixed & set(colored_mdp.coloring.relevantParameters(family.native, states))):
                wide = family.copy()
                wide.parameter_set_options(parameter, parameter_space.parameter_options(parameter))
                members += 1
                changed += selection_changes(colored_mdp, selected, choices, wide)
        assert members > 0
        assert changed / members > 0.5

    def test_thresholds_of_unchosen_variables_are_never_relevant(self, project, depth):
        """For a full assignment, the only things that matter are the decisions and thresholds actually tested and the leaves reached.

        The threshold of a variable an inner node does not branch on is never one of them -- exactly what the static per-state supports, which name every
        parameter of the tree, cannot say.
        """
        colored_mdp = tree_colored_mdp(project, depth)
        parameter_space = colored_mdp.parameter_space
        rng = random.Random(200 + depth)
        inner_nodes = colored_mdp.feature_info.decision_tree.collect_nonterminals()
        for _ in range(10):
            assignment, _fixed = random_family(rng, parameter_space, 1.0)
            _selected, states, _choices = reachable(colored_mdp, assignment)
            relevant = set(colored_mdp.coloring.relevantParameters(assignment.native, states))
            for node in inner_nodes:
                decision, *thresholds = node.parameters
                chosen_variable = assignment.parameter_options(decision)[0]
                for variable, threshold in enumerate(thresholds):
                    if variable != chosen_variable:
                        assert threshold not in relevant


@pytest.mark.parametrize("project", LIFTED)
class TestOnLiftedStandardColorings:
    def test_widening_outside_the_relevant_parameters_keeps_the_selection(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        check_widening_keeps_the_selection(general_colored_mdp(colored_mdp), seed=3, families=25, widenings=4, fixed_probability=0.5)

    def test_the_relevant_parameters_are_within_the_static_supports(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        general = general_colored_mdp(colored_mdp)
        supports = general.coloring.getStateToHoles()
        rng = random.Random(4)
        for _ in range(15):
            family, _fixed = random_family(rng, general.parameter_space, 0.5)
            _selected, states, _choices = reachable(general, family)
            static = set().union(*(set(supports[state]) for state in states))
            assert set(general.coloring.relevantParameters(family.native, states)) <= static


def two_parameter_coloring(state_is_relevant=None, num_states=1):
    """num_states states with two choices each, colored p == 0 and p == 1 (well-formed: exactly one is enabled), over parameters p, q with options {0, 1}."""
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    parameter_space.add_parameter("p", [0, 1])
    parameter_space.add_parameter("q", [0, 1])
    builder = paynt.utils.coloring_builder.ColoringBuilder([2 * state for state in range(num_states + 1)], 2, state_is_relevant=state_is_relevant)
    p, _q = builder.parameters
    for state in range(num_states):
        builder.color(2 * state, builder.template(p == 0))
        builder.color(2 * state + 1, builder.template(p == 1))
    return parameter_space, builder.build()


class TestSmallColorings:
    def test_only_the_parameters_that_exclude_a_choice_are_relevant(self):
        parameter_space, coloring = two_parameter_coloring()
        family = parameter_space.copy()
        family.parameter_set_options(0, [0])  # p == 1 is FALSE now, p == 0 is TRUE
        family.parameter_set_options(1, [1])  # q appears in no color at all
        assert sorted(coloring.relevantParameters(family.native, [0])) == [0]

    def test_nothing_is_relevant_while_every_choice_is_still_possible(self):
        parameter_space, coloring = two_parameter_coloring()
        assert sorted(coloring.relevantParameters(parameter_space.native, [0])) == []

    def test_irrelevant_states_are_skipped(self):
        relevant = stormpy.storage.BitVector(2, True)
        relevant.set(1, False)
        parameter_space, coloring = two_parameter_coloring(state_is_relevant=relevant, num_states=2)
        family = parameter_space.copy()
        family.parameter_set_options(0, [0])
        assert sorted(coloring.relevantParameters(family.native, [0])) == [0]
        assert sorted(coloring.relevantParameters(family.native, [1])) == []

    def test_exact_mode_keeps_the_support_of_a_color_it_excluded_without_a_kleene_reason(self):
        """P == q and p == 1 is undecided for Kleene evaluation once q is fixed to 0 (p is free), yet unsatisfiable: only the exact check excludes the choice it
        colors, and widening q would bring it back -- so q has to be relevant even though no color is Kleene-FALSE."""
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        parameter_space.add_parameter("p", [0, 1])
        parameter_space.add_parameter("q", [0, 1])
        builder = paynt.utils.coloring_builder.ColoringBuilder([0, 2], 2)
        p, q = builder.parameters
        formula = z3.And(p == q, p == 1)
        builder.color(0, builder.template(formula))
        builder.color(1, builder.template(z3.Not(formula)))
        builder.check_definition(parameter_space)
        coloring = builder.build()

        family = parameter_space.copy()
        family.parameter_set_options(1, [0])
        wide = parameter_space.copy()

        def selected(fam):
            selection = coloring.selectCompatibleChoices(fam.native)
            return [choice for choice in range(2) if selection[choice]]

        assert selected(family) == [0, 1]  # Kleene evaluation cannot exclude the first choice
        assert sorted(coloring.relevantParameters(family.native, [0])) == []

        coloring.setExactMode(True)
        assert selected(family) == [1]  # the exact check does
        assert selected(wide) == [0, 1]  # ... but only while q is fixed
        assert 1 in coloring.relevantParameters(family.native, [0])
