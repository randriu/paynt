"""Fixture-based tests of SynthesizerSMPMC, mirroring tests/pomdp/test_pomdp_synthesizer.py's style:

instantiate the synthesizer directly, synthesize, and assert against a pinned answer -- plus cross-checks against AR/OneByOne on the same fixtures, since SMPMC
and those engines must agree on feasibility.
"""

from __future__ import annotations

import logging
import time

import pytest
import z3

import paynt.parser.sketch
import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.smpmc.synthesizer
import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_ar
import paynt.utils.timer

from helpers.helper import get_sketch_paths


class TestSmpmcOnThresholdProperty:
    def test_finds_a_satisfying_assignment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        result = synthesizer.run()
        assert result.success is True

    def test_agrees_with_ar_on_feasibility(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        smpmc_result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()

        # a genuinely separate load, not a reused fixture object: Synthesizer instances mutate
        # task.specification state (optimality threshold/reset), so AR and SMPMC must not share one task
        sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
        ar_factory, ar_task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_result = paynt.synthesizer.synthesizer.Synthesizer.for_method(ar_factory.build(), ar_task, "ar").run()

        assert smpmc_result.success == ar_result.success

    def test_infeasible_threshold_is_correctly_refuted(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        smpmc_tiny_task.specification.constraints[0].threshold = 1.5  # P>=1.5 can never hold
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success is False


class TestSmpmcOnOptimalityProperty:
    def test_finds_the_known_optimum(self, smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task)
        result = synthesizer.run()
        assert result.success is True
        assert result.value == pytest.approx(1.0)

    def test_a_seed_already_at_the_optimum_reports_no_improvement(self, smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task, caplog):
        """Regression test for a real point of confusion: --optimum-threshold means "confirm you can beat this", not "confirm this is achievable" -- seeded with
        the known optimum (1.0, itself the maximum possible probability, so nothing can beat it), the search correctly finds nothing, and used to do so in total
        silence.

        See Synthesizer.synthesize's elif branch.
        """
        with caplog.at_level(logging.INFO, logger="paynt.synthesizer.synthesizer"):
            synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task)
            result = synthesizer.run(optimum_threshold=1.0)
        assert result.success is False
        assert result.assignment is None
        assert any("no assignment improving the given --optimum-threshold 1.0 was found" in record.message for record in caplog.records)


class TestOptimumThresholdSeedLoggingIsEngineAgnostic:
    """The new log line lives in the generic Synthesizer.synthesize() driver, not in SMPMC's own code -- confirm AR (which only overrides synthesize_one, same
    as SMPMC) gets it too, on the same fixture."""

    def test_ar_also_reports_no_improvement_on_the_same_seed(self, smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task, caplog):
        with caplog.at_level(logging.INFO, logger="paynt.synthesizer.synthesizer"):
            result = paynt.synthesizer.synthesizer_ar.SynthesizerAR(smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task).run(optimum_threshold=1.0)
        assert result.success is False
        assert any("no assignment improving the given --optimum-threshold 1.0 was found" in record.message for record in caplog.records)


class TestSmpmcOnFamilyModel:
    """Mdp-family-avoid-8-2-easy is a feature_kind=="family" sketch: --method ar on it routes to PolicyTreeSynthesizer (a robust-by-construction game-

    abstraction algorithm solving a *different* problem), not plain existential search -- so the correct cross-check for Phase 1's plain-exists SMPMC is
    --method onebyone, which api.py also routes through the generic Synthesizer.for_method path.
    """

    def test_agrees_with_onebyone_on_feasibility(self, mdp_family_colored_mdp, mdp_family_task):
        smpmc_result = paynt.synthesizer.smpmc.SynthesizerSMPMC(mdp_family_colored_mdp, mdp_family_task).run()

        sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
        onebyone_factory, onebyone_task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        onebyone_result = paynt.synthesizer.synthesizer.Synthesizer.for_method(onebyone_factory.build(), onebyone_task, "onebyone").run()

        assert smpmc_result.success == onebyone_result.success


class RecordingSolver:
    """Stands in for a z3.Solver where only the parameters set on it matter."""

    def __init__(self):
        self.settings = []

    def set(self, name, value):
        self.settings.append((name, value))


class TestLimitSolverTime:
    """The time left is what Z3 is told to give up after: milliseconds, at least one, at most what an unsigned 32-bit number holds."""

    @staticmethod
    def timeout_set_for(smpmc_tiny_colored_mdp, smpmc_tiny_task, limit):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        synthesizer.synthesis_timer = paynt.utils.timer.Timer(limit)
        synthesizer.synthesis_timer.start()
        solver = RecordingSolver()
        synthesizer._limit_solver_time(solver)
        return solver.settings

    def test_passes_the_remaining_time_in_milliseconds(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        [(name, milliseconds)] = self.timeout_set_for(smpmc_tiny_colored_mdp, smpmc_tiny_task, 10)
        assert name == "timeout"
        assert 9000 < milliseconds <= 10000

    def test_sets_nothing_without_a_time_limit(self, smpmc_tiny_colored_mdp, smpmc_tiny_task, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        assert self.timeout_set_for(smpmc_tiny_colored_mdp, smpmc_tiny_task, None) == []

    def test_a_limit_already_reached_still_gives_z3_a_timeout(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """Zero would mean no timeout at all to Z3."""
        assert self.timeout_set_for(smpmc_tiny_colored_mdp, smpmc_tiny_task, -5) == [("timeout", 1)]

    def test_a_huge_limit_is_capped_to_what_z3_accepts(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        assert self.timeout_set_for(smpmc_tiny_colored_mdp, smpmc_tiny_task, 10**9) == [("timeout", 2**32 - 1)]


class TestTimeIsUp:
    """What the theory asks at every Z3 callback: is the time limit reached."""

    @staticmethod
    def synthesizer_with_limit(smpmc_tiny_colored_mdp, smpmc_tiny_task, limit):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        synthesizer.synthesis_timer = paynt.utils.timer.Timer(limit)
        synthesizer.synthesis_timer.start()
        return synthesizer

    def test_is_false_while_time_is_left(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        assert not self.synthesizer_with_limit(smpmc_tiny_colored_mdp, smpmc_tiny_task, 60)._time_is_up()

    def test_is_true_once_the_limit_is_reached(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        assert self.synthesizer_with_limit(smpmc_tiny_colored_mdp, smpmc_tiny_task, -1)._time_is_up()

    def test_is_false_without_a_time_limit(self, smpmc_tiny_colored_mdp, smpmc_tiny_task, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        assert not self.synthesizer_with_limit(smpmc_tiny_colored_mdp, smpmc_tiny_task, None)._time_is_up()

    def test_is_what_the_theory_asks(self, smpmc_tiny_colored_mdp, smpmc_tiny_task, monkeypatch):
        """The synthesizer hands it to the theory when it sets the search up."""
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task)
        created = []
        real_theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory

        class Recording(real_theory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        monkeypatch.setattr(paynt.synthesizer.smpmc.checker, "ColoredMdpTheory", Recording)
        synthesizer.synthesize()
        assert created and created[0].time_is_up == synthesizer._time_is_up


class TestSolverGivesUp:
    """Z3 answers `unknown` when it was stopped (by the timeout or an interrupt) and when it cannot decide the formula at all: the first is the time limit
    working as intended, the second a warning, and neither is an answer to report."""

    @staticmethod
    def give_up_with(monkeypatch, reason):
        class GivesUp(z3.Solver):
            def check(self, *assumptions):
                return z3.unknown

            def reason_unknown(self):
                return reason

        monkeypatch.setattr(z3, "Solver", GivesUp)

    @pytest.mark.parametrize("reason", ["timeout", "canceled", "interrupted"])
    def test_a_stopped_search_is_the_time_limit_and_not_a_warning(self, monkeypatch, caplog, smpmc_tiny_colored_mdp, smpmc_tiny_task, reason):
        self.give_up_with(monkeypatch, reason)
        with caplog.at_level(logging.INFO):
            result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success is False
        assert any("time limit reached" in record.message for record in caplog.records)
        assert not any("returned unknown" in record.message for record in caplog.records)

    def test_any_other_reason_is_a_warning_naming_it(self, monkeypatch, caplog, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        self.give_up_with(monkeypatch, "incomplete quantifiers")
        with caplog.at_level(logging.INFO):
            result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success is False
        assert [record.message for record in caplog.records if "returned unknown" in record.message] == ["SMPMC returned unknown: incomplete quantifiers"]
        assert not any("time limit reached" in record.message for record in caplog.records)

    def test_the_best_assignment_found_so_far_stands(self, monkeypatch, smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task):
        """An optimality search that finds an assignment, and then gets nothing but unknown, reports that assignment."""
        real_check = z3.Solver.check
        calls = []

        class GivesUpAfterTheFirstAnswer(z3.Solver):
            def check(self, *assumptions):
                calls.append(1)
                return real_check(self, *assumptions) if len(calls) == 1 else z3.unknown

            def reason_unknown(self):
                return "incomplete quantifiers" if len(calls) > 1 else super().reason_unknown()

        monkeypatch.setattr(z3, "Solver", GivesUpAfterTheFirstAnswer)
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality_colored_mdp, smpmc_tiny_optimality_task).run()
        assert len(calls) >= 2
        assert result.success is True
        assert result.value is not None


class TestSmpmcTimeLimit:
    """A time limit stops the solver itself: one solver.check() is Z3's whole search, theory calls included, so it can run for minutes, and testing the limit
    only between two checks is not enough.

    Each theory call is slowed down, so that the searches below take many seconds whatever the machine.
    """

    @pytest.fixture(autouse=True)
    def slow_theory(self, monkeypatch):
        check = paynt.synthesizer.smpmc.checker.ColoredMdpTheory.check

        def slow_check(self, fixed, polarity):
            time.sleep(0.002)
            return check(self, fixed, polarity)

        monkeypatch.setattr(paynt.synthesizer.smpmc.checker.ColoredMdpTheory, "check", slow_check)

    @staticmethod
    def maze_synthesizer():
        """SMPMC on the decision trees of depth 2 of tests/dt-maze (Rmin): the optimum, 20.948..., takes some 20 000 theory calls to find and prove, a first
        tree only a couple."""
        sketch_path, props_path = get_sketch_paths("tests/dt-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        return paynt.synthesizer.smpmc.SynthesizerSMPMC(factory.reset_tree(2, general=True), task)

    def test_the_limit_ends_a_single_check_that_would_run_for_much_longer(self, caplog):
        """Seeded just below the optimum, nothing is left to find and the whole search is one solver.check() ending in unsat: some 9 500 theory calls, half a
        minute here without the limit."""
        synthesizer = self.maze_synthesizer()
        started = time.perf_counter()
        with caplog.at_level(logging.INFO):
            synthesizer.synthesize(optimum_threshold=20.9, timeout=0.5)
        assert time.perf_counter() - started < 5
        assert any("time limit reached" in record.message for record in caplog.records)
        assert not any("returned unknown" in record.message for record in caplog.records)

    def test_the_theory_stops_the_search_without_z3s_timeout(self, caplog, monkeypatch):
        """Z3's own timeout is the backstop and fires late; the theory interrupts at the first callback past the limit."""
        monkeypatch.setattr(paynt.synthesizer.smpmc.SynthesizerSMPMC, "_limit_solver_time", lambda self, solver: None)
        synthesizer = self.maze_synthesizer()
        started = time.perf_counter()
        with caplog.at_level(logging.INFO):
            synthesizer.synthesize(optimum_threshold=20.9, timeout=0.5)
        assert time.perf_counter() - started < 3
        assert any("time limit reached" in record.message for record in caplog.records)
        assert not any("returned unknown" in record.message for record in caplog.records)

    def test_a_search_stopped_by_the_limit_leaves_the_next_one_alone(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """A Z3_interrupt sent while no check() is running would leave the context cancelled, so that the next solver with a propagator answered sat at once,
        with an empty model."""
        self.maze_synthesizer().synthesize(optimum_threshold=20.9, timeout=0.3)
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success is True

    def test_the_best_assignment_found_before_the_limit_stands(self):
        synthesizer = self.maze_synthesizer()
        synthesizer.synthesize(keep_optimum=True, timeout=1)
        assert synthesizer.best_assignment is not None
        # Rmin: a tree worse than the optimum, since the search was cut short
        assert synthesizer.best_assignment_value > 20.95

    def test_a_robust_search_is_stopped_by_the_limit_too(self, rocks_colored_mdp, rocks_task):
        """There Z3 also runs the theory in quantifier-instantiation sub-contexts, and starts over with a new solver after each improvement."""
        rocks_task.constraint_name = "exists_forall"
        rocks_task.forall_pattern = None
        rocks_task.verify_robust = False
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, rocks_task)
        started = time.perf_counter()
        synthesizer.synthesize(timeout=0.5)
        assert time.perf_counter() - started < 5
