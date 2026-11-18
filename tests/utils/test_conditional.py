"""Conditional properties P[F target || F condition] on models/tests/conditional.

In sketch.templ, x=0 has value 0 and x=1 has 1/2; a policy that sets x anew after the condition reaches 1 (x=0, then x=1).

In undefined.templ, h=0 never reaches the condition, h=1 has value 1/4 and h=2 has 3/4; with h@condition=0, every path that reaches the condition reaches the
goal.
"""

from __future__ import annotations

import itertools
import math
import random
import re

import pytest
import stormpy

import paynt.api
import paynt.model.model
import paynt.specification.property
import paynt.synthesizer.synthesizer
import paynt.utils.conditional

from tests.helpers.models import get_sketch_paths, load_sketch
from tests.helpers.reference import Reference
from tests.soundness.test_random_sketches import random_sketch

PROJECT = "tests/conditional"
METHODS = ["onebyone", "ar", "smpmc"]
CONDITIONAL = 'F "goal" || F "condition"'


def props_file(tmp_path, *properties):
    path = tmp_path / "case.props"
    path.write_text("\n".join(properties) + "\n")
    return str(path)


def run(method, sketch_name="sketch.templ", props_path=None, conditional_parameters="shared", **kwargs):
    sketch_path, default_props_path = get_sketch_paths(PROJECT, sketch_name)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path or default_props_path, **kwargs)
    task.conditional_parameters = conditional_parameters
    return paynt.api.get_synthesizer(factory, task, method).run()


def state_variable(colored_mdp, design, state):
    """The value of the variable s in a state of a design."""
    valuation = str(colored_mdp.underlying_mdp.state_valuations.to_string(design.underlying_mdp_state_map[state]))
    return int(re.search(r"\bs=(\d+)", valuation).group(1))


def root_mdp(sketch_name, props_path):
    factory, task = load_sketch(PROJECT, sketch_name, props_path)
    colored_mdp = factory.build()
    return colored_mdp, colored_mdp.build(colored_mdp.parameter_space)[0], task.specification.all_properties()[0]


@pytest.mark.parametrize("method", METHODS)
class TestSharedParameters:
    def test_the_optimum_is_that_of_a_design(self, method):
        result = run(method)
        assert str(result.assignment) == "x=1"
        assert result.value == pytest.approx(0.5)

    def test_a_design_that_never_reaches_the_condition_is_not_the_least(self, method, tmp_path):
        result = run(method, "undefined.templ", props_file(tmp_path, f"Pmin=? [{CONDITIONAL}]"))
        assert str(result.assignment) == "h=1"
        assert result.value == pytest.approx(0.25)

    def test_a_design_that_never_reaches_the_condition_satisfies_no_constraint(self, method, tmp_path):
        result = run(method, "undefined.templ", props_file(tmp_path, f"P<=0.5 [{CONDITIONAL}]"))
        assert str(result.assignment) == "h=1"


@pytest.mark.parametrize("method", METHODS)
class TestPerPhaseParameters:
    def test_the_parameters_are_set_anew_after_the_condition(self, method):
        result = run(method, conditional_parameters="per-phase")
        assert str(result.assignment) == "x=0, x@condition=1"
        assert result.value == pytest.approx(1)

    def test_after_the_condition_every_path_reaches_the_goal(self, method, tmp_path):
        result = run(method, "undefined.templ", props_file(tmp_path, f"Pmax=? [{CONDITIONAL}]"), conditional_parameters="per-phase")
        assert result.value == pytest.approx(1)


@pytest.mark.parametrize("sketch_name", ["sketch.templ", "undefined.templ"])
def test_per_phase_designs_that_repeat_the_parameters_have_the_values_of_the_shared_ones(sketch_name, tmp_path):
    """The unfolded model keeps the labels and the rewards: every property has the value of the reference."""
    # the constraint first, as in Specification.all_properties
    properties = ['R{"steps"}<=2 [F "goal"]', f"Pmax=? [{CONDITIONAL}]"] if sketch_name == "undefined.templ" else [f"Pmax=? [{CONDITIONAL}]"]
    sketch_path, _ = get_sketch_paths(PROJECT, sketch_name)
    reference = Reference(sketch_path, properties)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_file(tmp_path, *properties))
    unfolded = paynt.utils.conditional.unfold_phases(factory.build(), task.specification)
    space = unfolded.parameter_space
    for design in reference.designs:
        assignment = space.copy()
        for parameter in range(space.num_parameters):
            hole = space.parameter_name(parameter).split("@")[0]
            label = dict(design.options)[hole]
            assignment.parameter_set_options(parameter, [space.parameter_to_option_labels[parameter].index(label)])
        model = unfolded.build_assignment(assignment)
        for prop, expected in zip(task.specification.all_properties(), design.values, strict=True):
            value = model.model_check_property(prop).value
            assert value == (None if expected is None else pytest.approx(float(expected))), f"{design}: {prop}"


def test_the_copies_of_a_parameter_have_its_options(tmp_path):
    sketch_path, _ = get_sketch_paths(PROJECT, "undefined.templ")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_file(tmp_path, f"P>=0.5 [{CONDITIONAL}]"))
    unfolded = paynt.utils.conditional.unfold_phases(factory.build(), task.specification)
    assert str(unfolded.parameter_space) == "h: {0,1,2}, h@condition: {0,1,2}, h@target: {0,1,2}"


class TestModelChecking:
    def test_the_value_of_a_design_that_never_reaches_the_condition_is_undefined(self, tmp_path):
        colored_mdp, _mdp, prop = root_mdp("undefined.templ", props_file(tmp_path, f"Pmax=? [{CONDITIONAL}]"))
        result = colored_mdp.build_assignment(colored_mdp.parameter_space.assume_parameter_options_copy(0, [0])).model_check_property(prop)
        assert (result.value, result.sat, result.improves_optimum, str(result)) == (None, False, False, "undefined")

    def test_the_worst_case_is_undefined_if_a_design_avoids_the_condition(self, tmp_path):
        _colored_mdp, mdp, prop = root_mdp("undefined.templ", props_file(tmp_path, f"P>=0.1 [{CONDITIONAL}]"))
        assert mdp.model_check_property(prop).sat
        assert str(mdp.model_check_property(prop, alt=True)) == "undefined"

    def test_a_constraint_is_decided_on_an_mdp_without_its_value(self, tmp_path):
        _colored_mdp, mdp, prop = root_mdp("sketch.templ", props_file(tmp_path, f"P>=0.9 [{CONDITIONAL}]"))
        result = mdp.model_check_property(prop)
        assert (result.value, result.sat, str(result)) == (None, True, "satisfied")
        assert result.result.has_scheduler()

    @pytest.mark.parametrize("use_exact", [False, True])
    def test_a_design_has_the_value_of_every_state(self, use_exact, tmp_path):
        """In undefined.templ with h=2, s=3 and s=4 never reach the condition: Storm's value for them is infinity."""
        sketch_path, _ = get_sketch_paths(PROJECT, "undefined.templ")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_file(tmp_path, f"Pmax=? [{CONDITIONAL}]"), use_exact=use_exact)
        colored_mdp = factory.build()
        design = colored_mdp.build_assignment(colored_mdp.parameter_space.assume_parameter_options_copy(0, [2]))
        result = design.model_check_property(task.specification.all_properties()[0]).result
        values = {state_variable(colored_mdp, design, state): result.get_value(state) for state in range(design.model.nr_states)}
        expected = {0: 0.75, 1: 0.5, 2: 1, 3: math.inf, 4: math.inf}
        assert values == ({s: stormpy.Rational(str(v)) if v != math.inf else v for s, v in expected.items()} if use_exact else pytest.approx(expected))

    def test_a_constraint_is_decided_for_an_initial_state_other_than_0(self):
        """In initial-state-2.drn, the action a of the initial state, 2, reaches the goal after the condition."""
        model = stormpy.build_model_from_drn(get_sketch_paths(PROJECT, "initial-state-2.drn")[0])
        prop = paynt.specification.property.construct_specification(stormpy.parse_properties(f"P>=0.5 [{CONDITIONAL}]")).constraints[0]
        result = paynt.utils.conditional.model_check(paynt.model.model.Mdp(model), prop)
        assert (result.value, result.sat) == (None, True)

    def test_a_design_has_a_value(self, tmp_path):
        colored_mdp, _mdp, prop = root_mdp("sketch.templ", props_file(tmp_path, f"P>=0.9 [{CONDITIONAL}]"))
        result = colored_mdp.build_assignment(colored_mdp.parameter_space.assume_parameter_options_copy(0, [1])).model_check_property(prop)
        assert (result.value, result.sat) == (pytest.approx(0.5), False)

    @pytest.mark.parametrize("algorithm", list(paynt.specification.property.CONDITIONAL_ALGORITHMS))
    def test_every_algorithm_finds_the_optimum(self, algorithm):
        assert run("ar", conditional_algorithm=algorithm).value == pytest.approx(0.5)
        environment = paynt.specification.property.Property.environment
        assert environment.modelchecker().conditional().algorithm == paynt.specification.property.CONDITIONAL_ALGORITHMS[algorithm]

    def test_exact_synthesis_gives_the_exact_optimum(self):
        assert run("ar", use_exact=True).value == stormpy.Rational("1/2")


class TestSchedulerSelection:
    """Storm's best scheduler of sketch.templ takes x=0 in s=0 and x=1 in s=1, after the condition."""

    def test_it_takes_both_options_of_x(self):
        colored_mdp, mdp, prop = root_mdp("sketch.templ", "sketch.props")
        scheduler = mdp.model_check_property(prop).result.get_scheduler()
        assert colored_mdp.scheduler_selection(mdp, scheduler, prop) == [[0, 1]]

    def test_a_state_that_reaches_the_condition_almost_surely_takes_the_choice_that_attains_the_value(self, tmp_path):
        """In almost-sure-condition.templ, Storm's scheduler for the least value takes h=0 in s=0 (value 1) and keeps h=2 (value 1/2) for after the
        condition."""
        colored_mdp, mdp, prop = root_mdp("almost-sure-condition.templ", props_file(tmp_path, f"Pmin=? [{CONDITIONAL}]"))
        scheduler = mdp.model_check_property(prop).result.get_scheduler()
        assert colored_mdp.scheduler_selection(mdp, scheduler, prop) == [[2]]

    def test_the_score_of_x_is_the_distance_of_s_1(self):
        colored_mdp, mdp, prop = root_mdp("sketch.templ", "sketch.props")
        scheduler = mdp.model_check_property(prop).result.get_scheduler()
        assert paynt.utils.conditional.backward_scores(colored_mdp, mdp, prop, scheduler, {0: [0, 1]}) == {0: 1}


class TestRefused:
    @pytest.mark.parametrize("method", ["cegis", "hybrid"])
    def test_by_counterexample_methods(self, method):
        with pytest.raises(NotImplementedError, match="use --method ar, onebyone or smpmc"):
            run(method)

    def test_by_robust_synthesis(self):
        factory, task = load_sketch(PROJECT)
        task.constraint_name, task.forall_pattern = "exists_forall", "x"
        with pytest.raises(NotImplementedError, match="robust synthesis"):
            paynt.api.get_synthesizer(factory, task, "smpmc")

    def test_over_an_mdp_family(self, tmp_path):
        factory, task = load_sketch("tests/mdp-family-correlated", props_name=props_file(tmp_path, 'Pmax=? [F "goal" || F "goal"]'))
        with pytest.raises(NotImplementedError, match="plain sketches only"):
            paynt.api.get_synthesizer(factory, task, "onebyone")

    @pytest.mark.parametrize("text", [f'R{{"steps"}}max=? [{CONDITIONAL}]', 'P>=0.5 [F<=3 "goal" || F "condition"]'])
    def test_other_than_a_conditional_reachability_probability(self, text, tmp_path):
        with pytest.raises(NotImplementedError, match="only conditional properties supported"):
            load_sketch(PROJECT, "undefined.templ", props_file(tmp_path, text))

    def test_where_a_target_is_needed(self):
        _factory, task = load_sketch(PROJECT)
        with pytest.raises(NotImplementedError, match="conditional property"):
            task.specification.all_properties()[0].get_target_label()

    def test_per_phase_parameters_without_a_conditional_property(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one conditional property"):
            run("ar", props_path=props_file(tmp_path, 'Pmax=? [F "goal"]'), conditional_parameters="per-phase")


@pytest.mark.parametrize("seed", range(10))
def test_every_state_of_a_design_has_the_value_of_the_design_started_there(seed, tmp_path):
    """Checked against Storm's procedure for MDPs, on the design as an MDP with its initial state moved there."""
    text = random_sketch(random.Random(seed))
    sketch_path = tmp_path / "sketch.templ"
    sketch_path.write_text(text)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(str(sketch_path), props_file(tmp_path, f"Pmax=? [{CONDITIONAL}]"))
    colored_mdp = factory.build()
    prop = task.specification.all_properties()[0]
    space = colored_mdp.parameter_space
    for combination in itertools.product(*[space.parameter_options(parameter) for parameter in range(space.num_parameters)]):
        assignment = space.assume_options_copy([[option] for option in combination])
        try:
            design = colored_mdp.build_assignment(assignment)
        except paynt.utils.error_handling.DeadlockedDesigns:
            continue
        result = design.model_check_property(prop).result
        if result is None:
            # no state of the design reaches the condition from the initial state on
            continue
        constants = {space.parameter_name(parameter): space.parameter_to_option_labels[parameter][option] for parameter, option in enumerate(combination)}
        for state in range(design.model.nr_states):
            start = state_variable(colored_mdp, design, state)
            assert result.get_value(state) == pytest.approx(started_at(text, constants, start, tmp_path), abs=1e-4), f"{assignment}, s={start}"


def started_at(text, constants, start, tmp_path):
    """The conditional probability of the design with the given hole values started in s=start, by Storm's procedure for MDPs; infinity if undefined."""
    for name, value in constants.items():
        text = re.sub(rf"^hole int {name} in .*$", f"const int {name} = {value};", text, flags=re.M)
    text = re.sub(r"^dtmc$", "mdp", text, flags=re.M).replace("init 0;", f"init {start};")
    path = tmp_path / "started.prism"
    path.write_text(text)
    program = stormpy.parse_prism_program(str(path))
    reach, conditional = stormpy.parse_properties(f'Pmax=? [F "condition"]; Pmax=? [{CONDITIONAL}]', program)
    model = stormpy.build_model(program, [reach, conditional])
    if stormpy.model_checking(model, reach, only_initial_states=True).get_value(model.initial_states[0]) == 0:
        return math.inf
    return stormpy.model_checking(model, conditional, only_initial_states=True).get_value(model.initial_states[0])
