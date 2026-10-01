"""Phase 2 differential tests: decision_tree_coloring (paynt.dt.coloring_general) against ColoringSmt, on the two tracked DT fixtures (dt-orchard: Pmax; dt-
maze: Rmin).

reset_tree(general=True) is the only entry point exercised -- it, not decision_tree_coloring directly, is DtColoredMdpFactory's own contract.
"""

from __future__ import annotations

import random

import pytest
import stormpy.storage

import paynt.parser.sketch
import paynt.dt._utils
import paynt.dt.coloring_general
import paynt.dt.decision_tree
from helpers.helper import get_sketch_paths

PROJECTS = ["tests/dt-orchard", "tests/dt-maze"]
DEPTHS = [0, 1, 2]
# where the general coloring's areChoicesConsistent is checked against ColoringSmt's: its satisfiable queries cost seconds on dt-orchard at depth 1 and up to
# half a minute at depth 2 (ColoringSmt: well under a second), so that case is left out
CONSISTENCY_CASES = [("tests/dt-orchard", 0), ("tests/dt-orchard", 1), ("tests/dt-maze", 0), ("tests/dt-maze", 1), ("tests/dt-maze", 2)]


def load(project, depth):
    sketch_path, props_path = get_sketch_paths(project)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory.reset_tree(depth, general=True), factory.reset_tree(depth, general=False), task


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestHasGeneralColoring:
    def test_reset_tree_reports_which_coloring_it_built(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        assert cmdp_general.has_general_coloring is True
        assert cmdp_smt.has_general_coloring is False
        # feature_kind is unaffected: every DT code path still applies to the general-coloring variant
        assert cmdp_general.feature_kind == cmdp_smt.feature_kind == "dt"


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestParameterLayoutMatches:
    def test_get_family_info_matches_coloringsmt(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        assert cmdp_general.parameter_space.num_parameters == cmdp_smt.parameter_space.num_parameters
        for parameter in range(cmdp_general.parameter_space.num_parameters):
            assert cmdp_general.parameter_space.parameter_name(parameter) == cmdp_smt.parameter_space.parameter_name(parameter)
            assert cmdp_general.parameter_space.parameter_options(parameter) == cmdp_smt.parameter_space.parameter_options(parameter)


@pytest.mark.parametrize("project", PROJECTS)
@pytest.mark.parametrize("depth", DEPTHS)
class TestSelectionMatchesOnRandomFamilies:
    def test_full_family_selection_matches(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        general = set(cmdp_general.coloring.selectCompatibleChoices(parameter_space.native))
        smt = set(cmdp_smt.coloring.selectCompatibleChoices(cmdp_smt.parameter_space.native))
        assert general == smt

    def test_random_subfamily_selection_matches(self, project, depth):
        cmdp_general, cmdp_smt, _task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        rng = random.Random(depth)
        for _trial in range(10):
            sub_general = parameter_space.copy()
            sub_smt = cmdp_smt.parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                if len(options) > 1 and rng.random() < 0.6:
                    chosen = sorted(rng.sample(options, rng.randint(1, len(options))))
                    sub_general.parameter_set_options(parameter, chosen)
                    sub_smt.parameter_set_options(parameter, chosen)
            general = set(cmdp_general.coloring.selectCompatibleChoices(sub_general.native))
            smt = set(cmdp_smt.coloring.selectCompatibleChoices(sub_smt.native))
            assert general == smt

    def test_random_full_assignment_gives_the_same_specification_result(self, project, depth):
        cmdp_general, _cmdp_smt, task = load(project, depth)
        parameter_space = cmdp_general.parameter_space
        rng = random.Random(depth + 100)
        for _trial in range(5):
            assignment = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                assignment.parameter_set_options(parameter, [rng.choice(options)])
            dtmc = cmdp_general.build_assignment(assignment)
            result = dtmc.check_specification(task.specification)
            assert result.constraints_result is not None


def load_without_harmonization(project, depth):
    """The general coloring and ColoringSmt over the same tree, both asked only for the verdict of areChoicesConsistent."""
    factory = load_factory(project)
    return factory.reset_tree(depth, enable_harmonization=False, general=True), factory.reset_tree(depth, enable_harmonization=False, general=False)


def answer(colored_mdp, mask, family):
    """The answer of areChoicesConsistent for the choices in mask under family.

    ColoringSmt needs the family selected first, as a side effect: it reads which paths of the tree that selection enabled, and without it no choice constrains
    anything.
    """
    colored_mdp.build(family)
    return colored_mdp.are_choices_consistent(mask, family)


def check_answer(consistent, selection, parameter_space):
    """The answer of areChoicesConsistent without harmonization: one option per parameter, if consistent, else nothing."""
    if consistent:
        assert [len(options) for options in selection] == [1] * parameter_space.num_parameters
        for parameter, [option] in enumerate(selection):
            assert option in parameter_space.parameter_options(parameter)
    else:
        assert all(len(options) == 0 for options in selection)


@pytest.mark.parametrize(("project", "depth"), CONSISTENCY_CASES)
class TestAreChoicesConsistentMatchesColoringSmt:
    """Checks of areChoicesConsistent without harmonization -- whether some tree makes all the given choices and, if so, one such tree -- which is all that
    mapping a scheduler asks of it.

    The general coloring must answer as ColoringSmt does.
    """

    def test_the_choices_a_tree_makes_are_consistent(self, project, depth):
        cmdp_general, cmdp_smt = load_without_harmonization(project, depth)
        parameter_space = cmdp_general.parameter_space
        rng = random.Random(depth + 300)
        for _trial in range(3):
            tree = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                tree.parameter_set_options(parameter, [rng.choice(parameter_space.parameter_options(parameter))])
            mask = cmdp_general.coloring.selectCompatibleChoices(tree.native)  # what that tree makes: consistent by construction
            for colored_mdp in (cmdp_general, cmdp_smt):
                consistent, selection = answer(colored_mdp, mask, parameter_space)
                assert consistent
                check_answer(consistent, selection, parameter_space)
                # the tree found is not necessarily the one the choices came from, but it makes exactly those choices
                found = parameter_space.assume_options_copy(selection)
                assert set(colored_mdp.coloring.selectCompatibleChoices(found.native)) == set(mask)

    def test_the_verdict_on_a_perturbed_scheduler_is_the_same(self, project, depth):
        cmdp_general, cmdp_smt = load_without_harmonization(project, depth)
        parameter_space = cmdp_general.parameter_space
        underlying_mdp = cmdp_general.underlying_mdp
        row_groups = underlying_mdp.nondeterministic_choice_indices
        rng = random.Random(depth + 200)
        for _trial in range(5):
            # one choice per state, from a random subfamily's own compatible choices, then a handful of states swapped to another of their options: likely
            # inconsistent, but always a genuine choice-per-state mask
            subfamily = parameter_space.copy()
            for parameter in range(parameter_space.num_parameters):
                options = parameter_space.parameter_options(parameter)
                if len(options) > 1 and rng.random() < 0.5:
                    subfamily.parameter_set_options(parameter, sorted(rng.sample(options, rng.randint(1, len(options)))))
            _mdp, selected_choices = cmdp_general.build(subfamily)
            mask_choices = {next(c for c in range(row_groups[state], row_groups[state + 1]) if selected_choices[c]) for state in range(len(row_groups) - 1)}
            for _ in range(3):
                state = rng.randrange(len(row_groups) - 1)
                candidates = [c for c in range(row_groups[state], row_groups[state + 1]) if selected_choices[c]]
                if len(candidates) > 1:
                    mask_choices.discard(next(c for c in candidates if c in mask_choices))
                    mask_choices.add(rng.choice(candidates))
            mask = stormpy.storage.BitVector(underlying_mdp.nr_choices, False)
            for choice in mask_choices:
                mask.set(choice, True)

            general_consistent, general_selection = answer(cmdp_general, mask, subfamily)
            smt_consistent, smt_selection = answer(cmdp_smt, mask, subfamily)
            assert general_consistent == smt_consistent
            check_answer(general_consistent, general_selection, subfamily)
            check_answer(smt_consistent, smt_selection, subfamily)


def load_factory(project):
    sketch_path, props_path = get_sketch_paths(project)
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


def inconsistent_scheduler(colored_mdp):
    """The choices, at the states reachable from the initial state, of a scheduler that no tree of depth 0 (one action for every state) makes: a real action
    everywhere, but not the same one at every relevant state."""
    info = colored_mdp.feature_info
    underlying_mdp = colored_mdp.underlying_mdp
    row_groups = underlying_mdp.nondeterministic_choice_indices
    num_states = len(row_groups) - 1

    def is_real(choice):
        return info.action_labels[info.choice_to_action[choice]] != paynt.dt._utils.DONT_CARE_ACTION_LABEL

    real = [[c for c in range(row_groups[s], row_groups[s + 1]) if is_real(c)] or list(range(row_groups[s], row_groups[s + 1])) for s in range(num_states)]

    def reachable(policy):
        """(state, choice) for every state the policy reaches."""
        initial_state = underlying_mdp.initial_states[0]
        reached = {initial_state}
        frontier = [initial_state]
        result = []
        while frontier:
            state = frontier.pop()
            result.append((state, policy[state]))
            for destination in colored_mdp.choice_destinations[policy[state]]:
                if destination not in reached:
                    reached.add(destination)
                    frontier.append(destination)
        return result

    policy = [candidates[0] for candidates in real]
    for switched in range(num_states):
        if not (info.state_is_relevant[switched] and len(real[switched]) > 1):
            continue
        trial = policy.copy()
        trial[switched] = real[switched][-1]
        chosen = reachable(trial)
        if len({info.choice_to_action[c] for s, c in chosen if info.state_is_relevant[s] and is_real(c)}) > 1:
            mask = stormpy.storage.BitVector(underlying_mdp.nr_choices, False)
            for _state, choice in chosen:
                mask.set(choice, True)
            return mask
    raise AssertionError("the fixture has no such scheduler")


@pytest.mark.parametrize("project", PROJECTS)
class TestAnInconsistentSchedulerIsRejected:
    """No tree of depth 0 -- one action for every state -- makes a scheduler that takes two different real actions: both colorings say so, and neither proposes
    a split (harmonization is AR's, and AR is not implemented over the general coloring)."""

    def test_no_split_is_proposed(self, project):
        for colored_mdp in load_without_harmonization(project, 0):
            consistent, selection = answer(colored_mdp, inconsistent_scheduler(colored_mdp), colored_mdp.parameter_space)
            assert not consistent
            assert selection == [[]]  # depth 0 has one parameter, the leaf's action


def tree_coloring_inputs(num_actions, dont_care_action=0):
    """decision_tree_coloring's inputs for a depth-0 tree over one state that offers the actions 0 and 1 only, in a model of num_actions actions: all the others
    are unavailable there."""
    action_labels = [f"a{action}" for action in range(num_actions)]
    tree = paynt.dt.decision_tree.DecisionTree(action_labels, [])
    tree.set_depth(0)
    return {
        "row_groups": [0, 2],
        "choice_to_action": [0, 1],
        "dont_care_action": dont_care_action,
        "action_labels": action_labels,
        "variables": [],
        "relevant_state_valuations": [[]],
        "state_is_relevant_bv": stormpy.storage.BitVector(1, True),
        "decision_tree": tree,
    }


class TestManyActions:
    """The actions unavailable at a state are one signed 64-bit word (ColoringBuilder.in_bits), so with a don't-care action an unavailable action can have index
    62 at most; a model with more used to fail in pybind with an opaque TypeError."""

    def test_the_highest_action_that_fits_is_supported(self):
        coloring, parameter_info = paynt.dt.coloring_general.decision_tree_coloring(**tree_coloring_inputs(63))
        assert [name for _node, name, _kind in parameter_info] == ["A_0"]

    def test_one_more_action_is_refused_with_a_clear_error(self):
        with pytest.raises(ValueError, match=r"action 'a63' \(index 63 of 64\) is unavailable at some state: use --method ar"):
            paynt.dt.coloring_general.decision_tree_coloring(**tree_coloring_inputs(64))

    def test_the_dont_care_action_may_have_the_highest_index(self):
        """It exists at relevant states only, so it must not count as unavailable at an irrelevant one: 63 actions and the don't-care action after them are
        fine."""
        num_actions = 64
        relevant = stormpy.storage.BitVector(2, True)
        relevant.set(1, False)
        inputs = tree_coloring_inputs(num_actions, dont_care_action=num_actions - 1)
        inputs.update(row_groups=[0, 3, 4], choice_to_action=[0, 1, num_actions - 1, 0], relevant_state_valuations=[[], []], state_is_relevant_bv=relevant)
        paynt.dt.coloring_general.decision_tree_coloring(**inputs)

    def test_a_model_without_a_dont_care_action_has_no_such_limit(self):
        """No word of unavailable actions is built then."""
        num_actions = 70
        paynt.dt.coloring_general.decision_tree_coloring(**tree_coloring_inputs(num_actions, dont_care_action=num_actions))
