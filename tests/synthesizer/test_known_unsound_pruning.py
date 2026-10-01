"""Minimal benchmarks on which an engine wrongly reports that nothing satisfies a specification that something does: its pruning is unsound.

OneByOne model-checks every design, so it is the reference. The expected values are worked out by hand in the sketches
(models/tests/objective-optimum-violates-constraint and models/tests/max-reward-infinite-designs), and checked against OneByOne below, not taken from
it.

The engines that get a case wrong are marked as expected failures, strictly: the day a bug is fixed the test passes unexpectedly, which is a failure
that says the mark has to go. Both bugs are in the abstraction-refinement search (hybrid shares the first), and the second one is SMPMC's as well.

1. A constraint and an objective together. At the root of the search, the scheduler that is best for the objective alone is one single design ("the
lower bound is tight", SynthesizerAR.check_specification). When that design violates the constraint, the whole node is dropped, although it holds
designs that satisfy the constraint. The node has to be refined instead.

2. A maximizing reward (an objective, or a constraint `R>=x`). The MDP that holds all the designs has an infinite maximal expected reward as soon as
one design never reaches the goal: the maximum over the MDP is the supremum over all of its resolutions, and one of them has an infinite reward. That
is an upper bound of the family and says nothing about the designs that do reach the goal. But Property.result_valid rejects an infinite reward (right
for the reward of one design, which then never reaches the goal), so the bound is taken for an invalid result and the node is dropped. An infinite
bound has to be inconclusive.
"""

from __future__ import annotations

import math

import pytest

import paynt.parser.sketch
import paynt.synthesizer.synthesizer

from helpers.helper import get_sketch_paths, load_colored_mdp

OBJECTIVE_VIOLATES_CONSTRAINT = "tests/objective-optimum-violates-constraint"
MAX_REWARD = "tests/max-reward-infinite-designs"


def run(project, method, props_name="sketch.props"):
    sketch_path, props_path = get_sketch_paths(project, props_name=props_name)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, method).run()


def expected_failure(reason):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


class TestAConstraintAndAnObjectiveTogether:
    """SAFE=0 costs 1 and delivers with probability 0.5, SAFE=1 costs 2 and delivers with probability 0.99: with `P>=0.9 [F "delivered"]` and the cost to
    minimize, only SAFE=1 is admissible, and it costs 2."""

    def test_the_designs_are_what_the_sketch_says(self):
        colored_mdp, task = load_colored_mdp(OBJECTIVE_VIOLATES_CONSTRAINT)
        space = colored_mdp.parameter_space
        expected = {(0,): (0.5, 1.0), (1,): (0.99, 2.0)}
        for combination, (probability, cost) in expected.items():
            result = colored_mdp.build_assignment(space.construct_assignment(combination)).check_specification(task.specification)
            assert result.constraints_result.results[0].value == pytest.approx(probability)
            assert result.optimality_result.value == pytest.approx(cost)
            assert result.constraints_result.sat is (probability >= 0.9)

    @pytest.mark.parametrize("method", ["onebyone", "cegis"])
    def test_the_engines_that_are_right_find_safe_1_with_cost_2(self, method):
        result = run(OBJECTIVE_VIOLATES_CONSTRAINT, method)
        assert result.success
        assert result.value == pytest.approx(2.0)
        assert str(result.assignment) == "SAFE=1"

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param(
                "ar",
                marks=expected_failure("the node is dropped: the scheduler that is best for the objective alone is one design, which violates the constraint"),
            ),
            pytest.param("hybrid", marks=expected_failure("its abstraction refinement drops the node first, as AR does")),
        ],
    )
    def test_the_engines_that_search_by_abstraction_find_it_too(self, method):
        result = run(OBJECTIVE_VIOLATES_CONSTRAINT, method)
        assert result.success
        assert result.value == pytest.approx(2.0)
        assert str(result.assignment) == "SAFE=1"


class TestAMaximizingReward:
    """EXIT=1 reaches the goal after 3 steps in expectation, EXIT=0 never does (an infinite cost, which PAYNT counts as invalid): `R{"cost"}max=?` is 3 with
    EXIT=1, and `R{"cost"}>=2.5` holds for it."""

    def test_the_designs_are_what_the_sketch_says(self):
        colored_mdp, task = load_colored_mdp(MAX_REWARD)
        space = colored_mdp.parameter_space
        prop = task.get_property()
        values = {
            combination: colored_mdp.build_assignment(space.construct_assignment(combination)).model_check_property(prop).value for combination in [(0,), (1,)]
        }
        assert values[(0,)] == math.inf
        assert values[(1,)] == pytest.approx(3.0)
        assert not prop.result_valid(values[(0,)]) and prop.result_valid(values[(1,)])

    def test_the_mdp_of_the_family_has_an_infinite_maximum_though_a_design_is_feasible(self):
        """That is the bound which the engines mistake for an invalid result: the maximum over all the resolutions of the MDP, one of which never ends."""
        colored_mdp, task = load_colored_mdp(MAX_REWARD)
        prop = task.get_property()
        mdp, _ = colored_mdp.build(colored_mdp.parameter_space)
        assert mdp.model_check_property(prop).value == math.inf
        assert mdp.model_check_property(prop, alt=True).value == pytest.approx(3.0, abs=1e-3)

    def test_onebyone_finds_exit_1_with_cost_3(self):
        result = run(MAX_REWARD, "onebyone")
        assert result.success
        assert result.value == pytest.approx(3.0)
        assert str(result.assignment) == "EXIT=1"

    def test_onebyone_finds_that_the_lower_bound_on_the_cost_holds_for_exit_1(self):
        result = run(MAX_REWARD, "onebyone", props_name="lower.props")
        assert result.success
        assert str(result.assignment) == "EXIT=1"

    @pytest.mark.parametrize("method", ["cegis", "hybrid"])
    def test_cegis_and_hybrid_do_not_run_a_maximizing_reward(self, method):
        with pytest.raises(AssertionError, match="maximizing reward"):
            run(MAX_REWARD, method)

    @pytest.mark.parametrize("props_name", ["sketch.props", "lower.props"])
    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("ar", marks=expected_failure("the infinite maximum of the MDP of the family is taken for an invalid result")),
            pytest.param("smpmc", marks=expected_failure("the infinite maximum of the MDP of the family is taken for an invalid result: `viable` is refuted")),
        ],
    )
    def test_the_engines_that_bound_the_family_find_exit_1(self, method, props_name):
        result = run(MAX_REWARD, method, props_name=props_name)
        assert result.success
        assert str(result.assignment) == "EXIT=1"
        if props_name == "sketch.props":
            assert result.value == pytest.approx(3.0)
