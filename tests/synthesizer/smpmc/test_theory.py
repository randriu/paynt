"""Integration tests of ColoredMdpTheory (Algorithm 1 "CheckMDPs") against a real colored MDP and real Storm model-checking calls, on the smpmc-tiny fixture
(see models/tests/smpmc-tiny)."""

from __future__ import annotations

import logging

import pytest

import paynt.parser.sketch
import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.statistic

from helpers.helper import get_sketch_paths


class TestColoredMdpTheoryOnFullAssignments:
    """For a *complete* assignment (every parameter fixed), check()'s verdict has an unambiguous ground
    truth: build the concrete DTMC directly and compare. This sidesteps needing to hand-derive which
    option index corresponds to which sketch-level hole label."""

    def test_every_full_assignment_agrees_with_direct_model_checking(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)

        parameter_space = smpmc_tiny_colored_mdp.parameter_space
        checked_at_least_one_sat = False
        checked_at_least_one_unsat = False
        for combination in parameter_space.all_combinations():
            fixed = dict(enumerate(combination))
            assignment = parameter_space.construct_assignment(combination)
            dtmc = smpmc_tiny_colored_mdp.build_assignment(assignment)
            real_sat = dtmc.model_check_property(prop).sat
            checked_at_least_one_sat |= real_sat
            checked_at_least_one_unsat |= not real_sat

            # viable(fixed) should hold iff real_sat -- so asking for polarity=True must be inconclusive
            # (not refuted) when real_sat is True, and refuted when real_sat is False
            viable_result = theory.check(fixed, True)
            assert (viable_result is None) == real_sat, f"{fixed}: real_sat={real_sat}, viable check={viable_result}"

            # dually for the not-viable literal
            not_viable_result = theory.check(fixed, False)
            assert (not_viable_result is None) == (not real_sat), f"{fixed}: real_sat={real_sat}, not-viable check={not_viable_result}"

        # a meaningful test needs the fixture to actually contain both outcomes
        assert checked_at_least_one_sat
        assert checked_at_least_one_unsat

    def test_refutation_conflict_is_a_subset_of_fixed(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        parameter_space = smpmc_tiny_colored_mdp.parameter_space
        for combination in parameter_space.all_combinations():
            fixed = dict(enumerate(combination))
            for polarity in (True, False):
                result = theory.check(fixed, polarity)
                if result is not None:
                    assert set(result.conflict_parameters).issubset(fixed.keys())


class TestColoredMdpTheorySingletonEta:
    """Regression coverage for a real correctness bug found on models/tests/generic-maze: for a fully- fixed (singleton) eta, ColoredMdp.build() hands back an
    MDP-typed model even though it has exactly one choice per state, and solving that via minmax/policy iteration can disagree with the exact value
    build_assignment()'s DTMC conversion gives for the mathematically identical process -- by an amount that lands just past model_checking_precision.

    That was enough to make SynthesizerSMPMC's optimality loop treat a non-improving witness as still-viable and terminate with the wrong (much worse) optimum:
    real 24-hole run converged to 36001.06 instead of the true minimum around 8.13. check() now uses build_assignment() directly whenever eta.size == 1,
    sidestepping the disagreement rather than working around its symptom.
    """

    def test_a_full_assignment_is_checked_via_the_exact_dtmc_value_not_build(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)

        assignment = colored_mdp.parameter_space.pick_any()
        expected = colored_mdp.build_assignment(assignment).model_check_property(prop, alt=False).value

        fixed = {p: assignment.parameter_options(p)[0] for p in range(assignment.num_parameters)}
        # a threshold exactly at the expected value: if check() ever computed this via build() (MDP-typed,
        # minmax-solved) instead of build_assignment() (DTMC-typed, exact), the two values would disagree
        # by an amount straddling model_checking_precision, and this refutation query would come back as
        # inconclusive (None) instead of correctly refuted -- exactly the bug that let a non-improving
        # witness through in the real run.
        prop.update_optimum(expected)
        result = theory.check(fixed, polarity=True)

        assert result is not None, (
            "expected this witness to be refuted (its value cannot strictly improve on itself), "
            "but check() returned inconclusive -- the singleton-eta shortcut may have regressed"
        )


class TestColoredMdpTheoryOutOfRangeValues:
    """Regression test: an out-of-range option must be treated as inconclusive (letting the plain in_range() Boolean constraint reject it), not passed through
    to ColoredMdp.build() -- which previously crashed stormpy's submodel construction with a deadlock-state error for some out-of-range partial assignments.

    See checker.py's check() for the guard this tests.
    """

    def test_out_of_range_option_is_inconclusive_not_a_crash(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        # parameter 0 (x1) only has options [0,1,2,3] -- 9999 is never valid
        assert theory.check({0: 9999}, True) is None
        assert theory.check({0: 9999}, False) is None


class TestColoredMdpTheoryRelevantParameters:
    """relevant_parameters(eta, sub_mdp) is the union of the static per-state supports (coloring.getStateToHoles()) over the states of the sub-MDP C[eta]; it is
    computed with BitVector ORs and has to agree with the union of plain sets."""

    @staticmethod
    def union_of_sets(colored_mdp, sub_mdp):
        supports = colored_mdp.coloring.getStateToHoles()
        union = set()
        for state in sub_mdp.underlying_mdp_state_map:
            union.update(supports[state])
        return union

    @pytest.mark.parametrize("project", ["tests/smpmc-tiny", "tests/generic-maze", "tests/mdp-family-avoid-8-2-easy"])
    def test_agrees_with_the_union_of_the_per_state_supports(self, project):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        specification = task.specification
        prop = specification.optimality if specification.optimality is not None else specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        space = colored_mdp.parameter_space

        step = max(1, space.num_parameters // 6)
        for num_fixed in range(0, space.num_parameters, step):
            eta = space.copy()
            for parameter in range(num_fixed):
                eta.parameter_set_options(parameter, [space.parameter_options(parameter)[0]])
            sub_mdp, _choices = colored_mdp.build(eta)
            relevant = theory.relevant_parameters(eta, sub_mdp)
            assert isinstance(relevant, set)
            assert relevant == self.union_of_sets(colored_mdp, sub_mdp), f"{num_fixed} parameters fixed"

    def test_the_sub_mdp_of_a_smaller_space_depends_on_no_more_parameters(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        space = smpmc_tiny_colored_mdp.parameter_space
        eta = space.copy()
        sub_mdp, _choices = smpmc_tiny_colored_mdp.build(eta)
        whole = theory.relevant_parameters(eta, sub_mdp)
        assert whole <= set(range(space.num_parameters))
        eta.parameter_set_options(0, [space.parameter_options(0)[0]])
        sub_mdp, _choices = smpmc_tiny_colored_mdp.build(eta)
        assert theory.relevant_parameters(eta, sub_mdp) <= whole


class TestColoredMdpTheoryRewardConvergence:
    """Regression coverage for _model_check's value-iteration-to-policy-iteration escalation.

    models/tests/generic-maze is a genuine, naturally-occurring example of the failure mode this guards
    against: R{"steps"}max=?[F "goal"] is a large but finite value (~903.9 million) that value iteration
    converges to pathologically slowly -- capped VI (see Property.initialize's max_minmax_iterations)
    returns a wildly wrong snapshot (order 10^3-10^7, depending on the cap) long before it gets anywhere
    close, while the min direction of the very same property converges immediately with plain VI. This is
    not a synthetic scenario; it's the exact model that originally exposed the bug (see conversation).
    """

    def test_the_slow_converging_direction_escalates_to_the_correct_value(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)

        result = theory._model_check(sub_mdp, alt=True)  # the max (worst-case) direction

        assert theory._reward_use_policy_iteration.get(True) is True
        assert result.value == pytest.approx(903940953.7112827, rel=1e-6)

    def test_the_fast_converging_direction_does_not_escalate(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)

        result = theory._model_check(sub_mdp, alt=False)  # the min (best-case) direction

        assert theory._reward_use_policy_iteration.get(False, False) is False
        assert result.value == pytest.approx(6.889553051515894, rel=1e-6)

    def test_a_second_query_on_an_escalated_direction_skips_reverification(self):
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        prop = task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)

        first = theory._model_check(sub_mdp, alt=True)
        second = theory._model_check(sub_mdp, alt=True)

        assert second.value == first.value

    def test_probability_properties_are_unaffected(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """The escalation machinery only ever triggers for reward properties -- probability values are bounded in [0,1] and don't exhibit this failure mode, so
        a probability property should never end up flagged for policy iteration."""
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        sub_mdp, _selected_choices = smpmc_tiny_colored_mdp.build(smpmc_tiny_colored_mdp.parameter_space)

        theory._model_check(sub_mdp, alt=False)
        theory._model_check(sub_mdp, alt=True)

        assert theory._reward_convergence_verified == {}
        assert theory._reward_use_policy_iteration == {}


class TestColoredMdpTheoryRewardVerificationOnLargeModels:
    """A policy-iteration call is not affordable on a large sub-MDP (_PI_VERIFICATION_MAX_STATES), so value iteration's result is trusted there, once, with a
    warning -- and the verification is still done later, on a sub-MDP small enough: the direction is not marked verified."""

    @staticmethod
    def maze_theory_and_sub_mdp():
        sketch_path, props_path = get_sketch_paths("tests/generic-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.build()
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, task.specification.optimality)
        sub_mdp, _selected_choices = colored_mdp.build(colored_mdp.parameter_space)
        return theory, sub_mdp

    def test_the_verification_is_skipped_with_one_warning_per_direction(self, monkeypatch, caplog):
        monkeypatch.setattr(paynt.synthesizer.smpmc.checker, "_PI_VERIFICATION_MAX_STATES", 1)
        theory, sub_mdp = self.maze_theory_and_sub_mdp()
        with caplog.at_level(logging.WARNING, logger="paynt.synthesizer.smpmc.checker"):
            first = theory._model_check(sub_mdp, alt=True)
            second = theory._model_check(sub_mdp, alt=True)
            theory._model_check(sub_mdp, alt=False)
        warnings = [record.message for record in caplog.records if "skipping policy-iteration verification" in record.message]
        assert len(warnings) == 2  # once for each direction
        assert "the alt direction" in warnings[0] and "the primary direction" in warnings[1]
        assert f"{sub_mdp.model.nr_states} states" in warnings[0]
        assert second.value == first.value

    def test_the_direction_is_not_marked_verified_so_a_smaller_sub_mdp_is_still_verified(self, monkeypatch):
        monkeypatch.setattr(paynt.synthesizer.smpmc.checker, "_PI_VERIFICATION_MAX_STATES", 1)
        theory, sub_mdp = self.maze_theory_and_sub_mdp()
        theory._model_check(sub_mdp, alt=True)
        assert theory._reward_convergence_verified == {}
        assert theory._reward_use_policy_iteration == {}

        monkeypatch.setattr(paynt.synthesizer.smpmc.checker, "_PI_VERIFICATION_MAX_STATES", 10**6)
        result = theory._model_check(sub_mdp, alt=True)
        assert theory._reward_convergence_verified == {True: True}
        assert result.value == pytest.approx(903940953.7112827, rel=1e-6)


class TestColoredMdpTheoryCache:
    def test_repeated_query_is_served_from_cache_without_a_second_model_check(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        fixed = {0: 0, 1: 0, 2: 0}
        theory.check(fixed, True)
        calls_after_first = theory.mc_calls
        theory.check(fixed, True)
        assert theory.mc_calls == calls_after_first, "a repeated query should hit the cache, not call Storm again"

    def test_inconclusive_cache_entry_is_not_reused_across_an_epoch_change(self, smpmc_tiny_colored_mdp, smpmc_tiny_optimality_task):
        prop = smpmc_tiny_optimality_task.specification.optimality
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        fixed = {0: 0, 1: 0, 2: 0}
        theory.check(fixed, True)
        calls_after_first = theory.mc_calls
        theory.epoch += 1
        theory.check(fixed, True)
        assert theory.mc_calls == calls_after_first + 1, "an inconclusive verdict from a stale epoch must not be reused"


class TestColoredMdpTheoryStatisticWiring:
    """check() is optionally given the run's Statistic object, feeding SMPMC into the same
    iteration-counting/status-throttling/progress-percentage machinery AR and CEGIS already use (see
    Statistic.iteration/.status in paynt/synthesizer/statistic.py) -- this is what backs --method smpmc's
    periodic "progress X%, ..., iters = {...}, opt = ..." log line."""

    def _theory_with_real_stat(self, colored_mdp, task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task)
        synthesizer.stat = paynt.synthesizer.statistic.Statistic(synthesizer)
        synthesizer.explored = 0
        prop = task.specification.optimality if task.specification.has_optimality else task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, prop, synthesizer.stat)
        return synthesizer, theory

    def test_stat_is_optional(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """The default (no stat argument) must keep working -- test_propagator.py's fake-theory tests and every other test_theory.py test above construct
        ColoredMdpTheory this way."""
        prop = smpmc_tiny_task.specification.constraints[0]
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(smpmc_tiny_colored_mdp, prop)
        assert theory.stat is None
        # a full round of checks over every combination must not raise just because stat is absent
        for combination in smpmc_tiny_colored_mdp.parameter_space.all_combinations():
            theory.check(dict(enumerate(combination)), True)

    def test_a_model_check_is_recorded_as_a_dtmc_iteration(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        synthesizer, theory = self._theory_with_real_stat(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        assert synthesizer.stat.iterations_dtmc is None
        theory.check({0: 0, 1: 0, 2: 0}, True)  # a full assignment -> the singleton-eta/DTMC path
        assert synthesizer.stat.iterations_dtmc == 1
        assert synthesizer.stat.iterations_mdp is None

    def test_a_cache_hit_does_not_double_count_an_iteration(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        synthesizer, theory = self._theory_with_real_stat(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        fixed = {0: 0, 1: 0, 2: 0}
        theory.check(fixed, True)
        theory.check(fixed, True)
        assert synthesizer.stat.iterations_dtmc == 1

    def test_a_fresh_refutation_increments_explored_on_the_synthesizer(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        synthesizer, theory = self._theory_with_real_stat(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        assert synthesizer.explored == 0
        # {x1=1 (index 0)} alone is enough to refute viable for this sketch (P>=0.1[F "done"] fails for
        # every combination with x1's label 1 and x2's label 3 -- see test_theory.py's ground-truth sweep)
        result = theory.check({0: 0, 1: 0}, True)
        assert result is not None, "expected a refutation to set up this test meaningfully"
        assert synthesizer.explored > 0

    def test_a_cache_hit_does_not_double_count_explored(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        synthesizer, theory = self._theory_with_real_stat(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        fixed = {0: 0, 1: 0}
        theory.check(fixed, True)
        explored_after_first = synthesizer.explored
        assert explored_after_first > 0
        theory.check(fixed, True)
        assert synthesizer.explored == explored_after_first


class TestWorstCaseBounds:
    """A robust search on smpmc-tiny with x2 universally quantified: the policy x1=4, x3=8 (options 3 and 0) reaches "done" for both values of x2, so its worst
    case is 1.0."""

    policy = {0: 3, 2: 0}

    def _theory(self, colored_mdp, task):
        return paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, task.specification.constraints[0], forall_parameters=[1])

    def test_a_refuted_not_viable_literal_bounds_its_policy(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        assert theory.check(self.policy, False) is not None
        bound = theory.worst_case_bound(self.policy)
        assert bound is not None and bound.value == pytest.approx(1.0) and not bound.exact

    def test_a_refuted_full_assignment_makes_the_bound_exact(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.check({**self.policy, 1: 0}, False)
        bound = theory.worst_case_bound(self.policy)
        assert bound is not None and bound.exact

    def test_a_cached_refutation_bounds_its_policy_too(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.check(self.policy, False)
        theory._worst_case.clear()
        calls = theory.mc_calls
        theory.check({**self.policy, 1: 1}, False)
        assert theory.mc_calls == calls, "the query should have been answered by the cached refutation"
        assert theory.worst_case_bound(self.policy) is not None

    def test_a_viable_literal_does_not_bound_the_worst_case(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.check({0: 0, 1: 0, 2: 0}, True)
        assert theory.worst_case_bound({0: 0, 2: 0}) is None

    def test_a_cached_refutation_of_unknown_value_leaves_the_policy_unbounded(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """The environments of a policy are covered by refutations, and the worst of their values bounds the policy's worst case: one without a value (so, no
        bound on its region) means no bound at all for the policy."""
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.cache.insert_refuted({**self.policy, 1: 1}, False, [0, 1, 2], theory.epoch)
        calls = theory.mc_calls
        assert theory.check({**self.policy, 1: 1}, False) is not None
        assert theory.mc_calls == calls, "the query should have been answered by the cached refutation"
        assert theory.worst_case_bound(self.policy) is None

    def test_a_refutation_with_a_value_bounds_the_query_even_next_to_one_without(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """The refutation of x2 == 0 does not depend on x2 (the policy never reaches the state that reads it), so it refutes x2 == 1 as well, with its value."""
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        assert theory.check({**self.policy, 1: 0}, False) is not None
        theory.cache.insert_refuted({**self.policy, 1: 1}, False, [0, 1, 2], theory.epoch)
        assert theory.check({**self.policy, 1: 1}, False) is not None
        bound = theory.worst_case_bound(self.policy)
        assert bound is not None and bound.value == pytest.approx(1.0)

    def test_a_policy_without_a_bound_is_not_bounded_by_later_refutations_either(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.cache.insert_refuted({**self.policy, 1: 1}, False, [0, 1, 2], theory.epoch)
        theory.check({**self.policy, 1: 1}, False)
        theory.check({**self.policy, 1: 0}, False)
        assert theory.worst_case_bound(self.policy) is None

    def test_the_unknown_is_dropped_with_the_epoch_too(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.cache.insert_refuted({**self.policy, 1: 1}, False, [0, 1, 2], theory.epoch)
        theory.check({**self.policy, 1: 1}, False)
        theory.epoch += 1
        theory.check({**self.policy, 1: 0}, False)
        bound = theory.worst_case_bound(self.policy)
        assert bound is not None and bound.value == pytest.approx(1.0)

    def test_bounds_are_dropped_with_the_epoch(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """They cover the environments only under the threshold they were refuted against."""
        theory = self._theory(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        theory.check(self.policy, False)
        theory.epoch += 1
        assert theory.worst_case_bound(self.policy) is None
