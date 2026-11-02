"""The trips of models/tests/time-energy: two constraints on different rewards pull the search in different directions, so the engines have to split the space
between them."""

from __future__ import annotations

import itertools

import pytest

import paynt.synthesizer.synthesizer

from tests.helpers.models import load_colored_mdp, load_sketch

PROJECT = "tests/time-energy"
METHODS = ["onebyone", "ar", "cegis", "hybrid"]
# by hand: a stage takes 1, 2 or 3 hours, so the trips of 4 hours
FOUR_HOUR_TRIPS = {(0, 2), (1, 1), (2, 0)}


def run(method, props_name):
    factory, task = load_sketch(PROJECT, props_name=props_name)
    return paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, method).run()


def test_the_trips_are_what_the_sketch_says():
    colored_mdp, task = load_colored_mdp(PROJECT)
    for first, second in itertools.product(range(3), repeat=2):
        result = colored_mdp.build_assignment(colored_mdp.parameter_space.construct_assignment((first, second))).check_specification(task.specification)
        time, energy = (constraint.value for constraint in result.constraints_result.results)
        assert time == pytest.approx(first + second + 2)
        assert energy == pytest.approx(8 - time)
        assert result.constraints_result.sat is ((first, second) in FOUR_HOUR_TRIPS)


@pytest.mark.parametrize("method", METHODS)
def test_a_trip_of_4_hours_is_found(method):
    result = run(method, "sketch.props")
    assert result.success
    assert tuple(result.assignment.parameter_options(parameter)[0] for parameter in range(2)) in FOUR_HOUR_TRIPS


@pytest.mark.parametrize("method", METHODS)
def test_no_trip_takes_at_most_3_hours_and_4_units_of_energy(method):
    assert not run(method, "infeasible.props").success
