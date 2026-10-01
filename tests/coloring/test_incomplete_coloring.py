"""Incomplete colorings: a full parameter assignment may leave several choices enabled (see paynt.colored_mdp).

The induced model is then an MDP, and the specification is to hold for the best resolution of the choices left. The example is dt-maze with "the relevant states
with x == v all play the same action": one parameter per v, whose options are the actions all those states offer, and every other choice left uncolored -- a
problem that, with a parameter per state, would have 5^13 assignments, and here has between 5 and 625. The optimum is found by brute force, one MDP check per
assignment, and every engine must agree with it.
"""

from __future__ import annotations

import itertools

import pytest
import stormpy

import paynt.api
import paynt.colored_mdp
import paynt.dt._utils
import paynt.parameter_space.parameter_space
import paynt.parser.sketch
import paynt.synthesizer.smpmc
import paynt.synthesizer.synthesizer
import paynt.utils.coloring_builder

from helpers.helper import get_sketch_paths, load_colored_mdp

X_VALUES = [(2,), (0, 1, 3)]
# generic AR and Hybrid read (parameter, option) pairs back from the coloring, which only the standard one has
METHODS = {"standard": ["onebyone", "cegis", "ar", "hybrid", "smpmc"], "general": ["onebyone", "cegis", "smpmc"]}
CASES = [(kind, method) for kind, methods in METHODS.items() for method in methods]


def maze(props_path=None):
    sketch_path, default_props_path = get_sketch_paths("tests/dt-maze")
    return paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path or default_props_path)


def incomplete_maze_builder(x_values, props_path=None, all_actions=False):
    """(the builder, the parameter space, the underlying MDP, the task, the option labels): the relevant states with x == v share the action of one parameter,
    for every v in x_values.

    :param all_actions: give the parameters every action of the model as an option, including those some of the states do not offer
    """
    factory, task = maze(props_path)
    mdp = factory.underlying_mdp
    names, valuations = paynt.dt._utils.get_state_valuations(mdp)
    row_groups = mdp.nondeterministic_choice_indices
    groups = [[s for s in range(mdp.nr_states) if factory.state_is_relevant[s] and valuations[s][names.index("x")] == v] for v in x_values]
    if all_actions:
        options = list(range(len(factory.action_labels)))
    else:
        options = sorted(set.intersection(*[{factory.choice_to_action[c] for c in range(row_groups[s], row_groups[s + 1])} for group in groups for s in group]))
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for v in x_values:
        parameter_space.add_parameter(f"A_x{v}", [factory.action_labels[a] for a in options])

    builder = paynt.utils.coloring_builder.ColoringBuilder(row_groups, len(x_values))
    for parameter, group in zip(builder.parameters, groups, strict=True):
        if all_actions:
            # written by hand: the helper refuses an option that a state of the group does not offer
            action = builder.choice_column([options.index(a) if a in options else -1 for a in factory.choice_to_action])
            template = builder.template(parameter == action)
            for state in group:
                builder.color(range(row_groups[state], row_groups[state + 1]), template)
        else:
            builder.share_action(parameter, group, factory.choice_to_action, actions=options)
    return builder, parameter_space, mdp, task, [factory.action_labels[a] for a in options]


def incomplete_maze(x_values, kind="standard", props_path=None, all_actions=False):
    """(the colored MDP, its task, the option labels) of incomplete_maze_builder, with the standard or the general coloring."""
    builder, parameter_space, mdp, task, options = incomplete_maze_builder(x_values, props_path, all_actions)
    coloring = builder.build(parameter_space, general=(kind == "general"))
    return paynt.colored_mdp.ColoredMdp(mdp, parameter_space, coloring), task, options


def brute_force_optimum(colored_mdp, task):
    """The best value of the property over the assignments, each an MDP check, and the assignments that reach it."""
    space = colored_mdp.parameter_space
    prop = task.get_property()
    values = {}
    for combination in itertools.product(range(space.parameter_num_options(0)), repeat=space.num_parameters):
        assignment = space.copy()
        for parameter, option in enumerate(combination):
            assignment.parameter_set_options(parameter, [option])
        values[combination] = colored_mdp.build(assignment)[0].model_check_property(prop).value
    best = min(values.values()) if prop.minimizing else max(values.values())
    return best, [combination for combination, value in values.items() if value == pytest.approx(best, abs=1e-9)]


class TestFullAssignmentsOfAnIncompleteColoring:
    @pytest.mark.parametrize("kind", ["standard", "general"])
    def test_a_full_assignment_is_an_mdp(self, kind):
        colored_mdp, _task, _options = incomplete_maze((2,), kind)
        assignment = colored_mdp.parameter_space.copy()
        assignment.parameter_set_options(0, [1])
        model = colored_mdp.build_assignment(assignment)
        assert isinstance(model.model, stormpy.storage.SparseMdp)
        assert not model.is_deterministic

    def test_a_complete_coloring_still_gives_a_dtmc(self):
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        assignment = colored_mdp.parameter_space.copy()
        for parameter in range(assignment.num_parameters):
            assignment.parameter_set_options(parameter, [assignment.parameter_options(parameter)[0]])
        assert isinstance(colored_mdp.build_assignment(assignment).model, stormpy.storage.SparseDtmc)

    def test_it_violates_definition_2_and_the_relaxed_definition_holds(self):
        builder, parameter_space, _mdp, _task, _options = incomplete_maze_builder((0, 1, 3))
        with pytest.raises(AssertionError, match="Definition 2 violated"):
            builder.check_definition(parameter_space)
        builder.check_definition(parameter_space, complete=False)


@pytest.mark.parametrize("x_values", X_VALUES)
@pytest.mark.parametrize(("kind", "method"), CASES)
class TestEnginesFindTheOptimum:
    def test_the_optimum_of_the_property(self, kind, method, x_values):
        colored_mdp, task, _options = incomplete_maze(x_values, kind)
        optimum, _best = brute_force_optimum(colored_mdp, task)
        fresh_colored_mdp, fresh_task, _options = incomplete_maze(x_values, kind)
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(fresh_colored_mdp, fresh_task, method).run()
        assert result.success
        assert result.value == pytest.approx(optimum, abs=1e-6)


@pytest.mark.parametrize(("kind", "method"), CASES)
class TestEnginesDecideAThreshold:
    """The optimum for x groups (0, 1, 3) is 10.792."""

    @pytest.mark.parametrize(("threshold", "satisfiable"), [(11, True), (10, False)])
    def test_the_verdict(self, kind, method, threshold, satisfiable, tmp_path):
        props_path = tmp_path / "threshold.props"
        props_path.write_text(f'R{{"steps"}}<={threshold} [F ((x = 2) & (y = 0))]\n')
        colored_mdp, task, _options = incomplete_maze((0, 1, 3), kind, props_path=str(props_path))
        assert paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run().success is satisfiable


class TestSeveralPropertiesAreRefused:
    """No policy is shared between the properties, so each would get its own best one: the two targets below cannot both be reached with probability 1, yet each
    can alone."""

    @staticmethod
    def two_targets(props=None, tmp_path=None):
        """The colored MDP of tests/incomplete-two-targets: state 0 offers two choices that no parameter colors, and the only parameter colors the two choices
        of an unreachable state, so that every full assignment leaves state 0 open.

        The sketch is an MDP with a hole, which the parser takes for an MDP family (feature_kind "family"): there build_assignment keeps the MDP whatever the
        coloring, which is not what is tested here. Its parts make a generic colored MDP, as in the maze tests above.
        """
        sketch_path, props_path = get_sketch_paths("tests/incomplete-two-targets")
        if props is not None:
            props_path = str(tmp_path / "one.props")
            (tmp_path / "one.props").write_text(props)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        family = factory.build()
        assert family.feature_kind == "family"
        return paynt.colored_mdp.ColoredMdp(family.underlying_mdp, family.parameter_space, family.coloring), task

    @pytest.mark.parametrize("method", ["onebyone", "ar", "hybrid", "cegis"])
    def test_two_properties(self, method):
        colored_mdp, task = self.two_targets()
        with pytest.raises(NotImplementedError, match="cannot check 2 properties on a model that still has nondeterminism"):
            paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()

    def test_an_mdp_family_is_refused_too(self):
        """The same sketch, as the parser takes it: an MDP family, whose policy is the nondeterminism left by a full assignment.

        The policy tree does not support several properties either (SynthesisTask.get_property), so a family never has more than one.
        """
        sketch_path, props_path = get_sketch_paths("tests/incomplete-two-targets")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        assert factory.feature_kind == "family"
        with pytest.raises(NotImplementedError, match="cannot check 2 properties on a model that still has nondeterminism.*an MDP family"):
            paynt.api.get_synthesizer(factory, task, "onebyone").run()

    @pytest.mark.parametrize("method", ["onebyone", "ar", "hybrid", "cegis", "smpmc"])
    def test_either_property_alone_is_satisfiable(self, method, tmp_path):
        for target in ("A", "B"):
            colored_mdp, task = self.two_targets(f'P>=1 [F "{target}"]\n', tmp_path)
            assert paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run().success


class TestRobustSynthesisIsRefused:
    def test_an_incomplete_coloring(self):
        colored_mdp, task, _options = incomplete_maze((0, 1))
        task.constraint_name = "exists_forall"
        task.forall_pattern = "A_x1"
        task.verify_robust = False
        with pytest.raises(NotImplementedError, match="robust synthesis does not support an incomplete coloring"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run()


class TestAStateLeftWithoutAChoice:
    """The options of a hand-written parameter are the user's: one that a state of the group does not offer leaves that state with nothing to do."""

    def test_the_error_names_the_state(self):
        colored_mdp, _task, options = incomplete_maze((2,), all_actions=True)
        assignment = colored_mdp.parameter_space.copy()
        assignment.parameter_set_options(0, [options.index("place")])
        for build in (colored_mdp.build_assignment, lambda space: colored_mdp.build(space)):
            with pytest.raises(ValueError, match=r"the coloring enables no choice in state \d+, which is reachable"):
                build(assignment)

    @pytest.mark.parametrize("method", ["onebyone", "smpmc"])
    def test_an_engine_reports_it(self, method):
        colored_mdp, task, _options = incomplete_maze((2,), all_actions=True)
        with pytest.raises(ValueError, match="the coloring enables no choice in state"):
            paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()

    def test_the_offered_actions_alone_are_fine(self):
        colored_mdp, task, options = incomplete_maze((2,))
        assert "place" not in options
        assert paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, "onebyone").run().success


class TestTheResultCarriesTheSelectedChoices:
    """The choices of the underlying MDP the solution stands for: those its assignment enables and, where it leaves several, the ones the best policy takes --
    the uncolored ones included."""

    @staticmethod
    def check(colored_mdp, task, result, uncolored_are_taken):
        choices = result.selected_choices
        assert choices is not None
        assert set(choices) <= set(colored_mdp.coloring.selectCompatibleChoices(result.assignment.native))
        chain = colored_mdp.build_from_choice_mask(choices)
        assert chain.is_deterministic  # one choice for every state it reaches
        assert chain.model_check_property(task.get_property()).value == pytest.approx(result.value, rel=1e-3)
        if uncolored_are_taken:
            assert any(not colored_mdp.coloring.getChoiceToAssignment()[choice] for choice in choices)

    @pytest.mark.parametrize("method", ["onebyone", "ar", "smpmc", "cegis", "hybrid"])
    def test_an_incomplete_coloring(self, method):
        colored_mdp, task, _options = incomplete_maze((0, 1, 3))
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        self.check(colored_mdp, task, result, uncolored_are_taken=True)

    @pytest.mark.parametrize("method", ["onebyone", "ar", "smpmc", "cegis", "hybrid"])
    def test_a_complete_coloring_gives_the_choices_of_its_markov_chain(self, method):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
        assert result.success
        chain = colored_mdp.build_assignment(result.assignment)
        assert result.selected_choices.number_of_set_bits() == chain.model.nr_states
        assert set(result.selected_choices) == set(chain.underlying_mdp_choice_map)

    def test_the_general_coloring_too(self):
        colored_mdp, task, _options = incomplete_maze((0, 1, 3), "general")
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, "smpmc").run()
        self.check(colored_mdp, task, result, uncolored_are_taken=False)

    def test_none_without_a_solution(self, tmp_path):
        props_path = tmp_path / "unsat.props"
        props_path.write_text('R{"steps"}<=10 [F ((x = 2) & (y = 0))]\n')
        colored_mdp, task, _options = incomplete_maze((0, 1, 3), props_path=str(props_path))
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, "onebyone").run()
        assert not result.success
        assert result.selected_choices is None
