"""The reference of tests/helpers/reference.py against the answers worked out by hand in the sketches."""

from __future__ import annotations

import math
from fractions import Fraction

from tests.helpers.models import get_sketch_paths
from tests.helpers.reference import Reference, read_properties


def reference(project, props_name="sketch.props", sketch_name="sketch.templ", properties=None):
    sketch_path, props_path = get_sketch_paths(project, sketch_name, props_name)
    return Reference(sketch_path, properties or read_properties(props_path))


def table(reference):
    return {str(design): design.values for design in reference.designs}


def test_smpmc_tiny_has_three_designs_that_reach_done():
    feasible = reference("tests/smpmc-tiny").feasible
    assert {str(design) for design in feasible} == {"x1=1, x2=4, x3=8", "x1=4, x2=3, x3=8", "x1=4, x2=4, x3=8"}


def test_a_trip_takes_8_in_time_and_energy_together():
    trips = reference("tests/time-energy")
    for design in trips.designs:
        first, second = (int(option) for _, option in design.options)
        assert design.values == (first + second + 2, 6 - first - second)
    assert {str(design) for design in trips.feasible} == {"FIRST=0, SECOND=2", "FIRST=1, SECOND=1", "FIRST=2, SECOND=0"}


def test_only_the_safe_sender_delivers_often_enough():
    sender = reference("tests/objective-optimum-violates-constraint")
    assert table(sender) == {"SAFE=0": (Fraction(1, 2), 1), "SAFE=1": (Fraction(99, 100), 2)}
    assert sender.optimum == 2


def test_a_walker_that_never_reaches_the_goal_is_not_admissible():
    walker = reference("tests/max-reward-infinite-designs")
    assert table(walker) == {"EXIT=0": (math.inf,), "EXIT=1": (3,)}
    assert walker.optimum == 3


def test_the_boundaries_are_quarters_and_their_costs():
    properties = ['P=? [F "goal"]', 'R{"cost"}=? [F "done"]']
    designs = table(reference("tests/edge-cases", sketch_name="boundaries.templ", properties=properties))
    assert designs == {f"H={h}": (Fraction(h, 4), h) for h in (0, 1, 2, 4)}


def test_start_is_reached_at_once_and_nowhere_never():
    properties = ['P=? [F "start"]', 'R{"steps"}=? [F "start"]', 'P=? [F "nowhere"]', 'R{"steps"}=? [F "nowhere"]']
    designs = table(reference("tests/edge-cases", sketch_name="reach.templ", properties=properties))
    assert designs == {"H=0": (1, 0, 0, math.inf), "H=1": (1, 0, 0, math.inf)}


def test_a_probability_hole_keeps_its_options_as_written():
    designs = table(reference("tests/edge-cases", sketch_name="probability-hole.templ", properties=['P=? [F "goal"]']))
    assert designs == {"P=0.2": (Fraction(1, 5),), "P=0.5": (Fraction(1, 2),), "P=0.75": (Fraction(3, 4),)}
