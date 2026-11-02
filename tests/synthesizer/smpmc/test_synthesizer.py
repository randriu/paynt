"""SynthesizerSMPMC: the answers on smpmc-tiny are worked out by hand (see tests/README.md), the member of a family it finds is model-checked."""

from __future__ import annotations

import logging
import time

import pytest
import z3

import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_ar
import paynt.utils.timer

from tests.helpers.models import load_sketch


class TestSmpmcOnThresholdProperty:
    def test_finds_a_satisfying_assignment(self, smpmc_tiny):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task)
        result = synthesizer.run()
        assert result.success is True
        # by hand: "done" is reached with x1=4, or with x1=1 and x2=4
        assert str(result.assignment) in {"x1=4, x2=3, x3=8", "x1=4, x2=4, x3=8", "x1=1, x2=4, x3=8"}

    def test_infeasible_threshold_is_correctly_refuted(self, smpmc_tiny):
        smpmc_tiny.task.specification.constraints[0].threshold = 1.5  # P>=1.5 can never hold
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task).run()
        assert result.success is False


class TestSmpmcOnOptimalityProperty:
    def test_finds_the_known_optimum(self, smpmc_tiny_optimality):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality.colored_mdp, smpmc_tiny_optimality.task)
        result = synthesizer.run()
        assert result.success is True
        assert result.value == pytest.approx(1.0)

    def test_a_seed_already_at_the_optimum_reports_no_improvement(self, smpmc_tiny_optimality, caplog):
        """--optimum-threshold means "confirm you can beat this", not "confirm this is achievable": seeded with the known optimum (1.0, itself the maximum
        possible probability, so nothing can beat it), the search finds nothing and says so in a log line. See Synthesizer.synthesize's elif branch."""
        with caplog.at_level(logging.INFO, logger="paynt.synthesizer.synthesizer"):
            synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality.colored_mdp, smpmc_tiny_optimality.task)
            result = synthesizer.run(optimum_threshold=1.0)
        assert result.success is False
        assert result.assignment is None
        assert any("no assignment improving the given --optimum-threshold 1.0 was found" in record.message for record in caplog.records)


class TestOptimumThresholdSeedLoggingIsEngineAgnostic:
    """The log line lives in the generic Synthesizer.synthesize() driver, not in SMPMC's own code, so AR (which only overrides synthesize_one, like SMPMC) gets
    it too, on the same fixture."""

    def test_ar_also_reports_no_improvement_on_the_same_seed(self, smpmc_tiny_optimality, caplog):
        with caplog.at_level(logging.INFO, logger="paynt.synthesizer.synthesizer"):
            result = paynt.synthesizer.synthesizer_ar.SynthesizerAR(smpmc_tiny_optimality.colored_mdp, smpmc_tiny_optimality.task).run(optimum_threshold=1.0)
        assert result.success is False
        assert any("no assignment improving the given --optimum-threshold 1.0 was found" in record.message for record in caplog.records)


class TestSmpmcOnFamilyModel:
    """On a family, SMPMC searches for a member in which some policy satisfies the property."""

    def test_the_member_found_satisfies_the_property(self, mdp_family_avoid_8_2_easy):
        colored_mdp, task = mdp_family_avoid_8_2_easy.colored_mdp, mdp_family_avoid_8_2_easy.task
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run()
        assert result.success
        assert colored_mdp.build_assignment(result.assignment).check_specification(task.specification).constraints_result.sat


class RecordingSolver:
    """Stands in for a z3.Solver where only the parameters set on it matter."""

    def __init__(self):
        self.settings = []

    def set(self, name, value):
        self.settings.append((name, value))


class TestLimitSolverTime:
    """The time left is what Z3 is told to give up after: milliseconds, at least one, at most what an unsigned 32-bit number holds."""

    @staticmethod
    def timeout_set_for(smpmc_tiny, limit):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task)
        synthesizer.synthesis_timer = paynt.utils.timer.Timer(limit)
        synthesizer.synthesis_timer.start()
        solver = RecordingSolver()
        synthesizer._limit_solver_time(solver)
        return solver.settings

    def test_passes_the_remaining_time_in_milliseconds(self, smpmc_tiny):
        [(name, milliseconds)] = self.timeout_set_for(smpmc_tiny, 10)
        assert name == "timeout"
        assert 9000 < milliseconds <= 10000

    def test_sets_nothing_without_a_time_limit(self, smpmc_tiny, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        assert self.timeout_set_for(smpmc_tiny, None) == []

    def test_a_limit_already_reached_still_gives_z3_a_timeout(self, smpmc_tiny):
        """Zero would mean no timeout at all to Z3."""
        assert self.timeout_set_for(smpmc_tiny, -5) == [("timeout", 1)]

    def test_a_huge_limit_is_capped_to_what_z3_accepts(self, smpmc_tiny):
        assert self.timeout_set_for(smpmc_tiny, 10**9) == [("timeout", 2**32 - 1)]


class TestTimeIsUp:
    """What the theory asks at every Z3 callback: is the time limit reached."""

    @staticmethod
    def synthesizer_with_limit(smpmc_tiny, limit):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task)
        synthesizer.synthesis_timer = paynt.utils.timer.Timer(limit)
        synthesizer.synthesis_timer.start()
        return synthesizer

    def test_is_false_while_time_is_left(self, smpmc_tiny):
        assert not self.synthesizer_with_limit(smpmc_tiny, 60)._time_is_up()

    def test_is_true_once_the_limit_is_reached(self, smpmc_tiny):
        assert self.synthesizer_with_limit(smpmc_tiny, -1)._time_is_up()

    def test_is_false_without_a_time_limit(self, smpmc_tiny, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        assert not self.synthesizer_with_limit(smpmc_tiny, None)._time_is_up()

    def test_is_what_the_theory_asks(self, smpmc_tiny, monkeypatch):
        """The synthesizer hands it to the theory when it sets the search up."""
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task)
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
    def test_a_stopped_search_is_the_time_limit_and_not_a_warning(self, monkeypatch, caplog, smpmc_tiny, reason):
        self.give_up_with(monkeypatch, reason)
        with caplog.at_level(logging.INFO):
            result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task).run()
        assert result.success is False
        assert any("time limit reached" in record.message for record in caplog.records)
        assert not any("returned unknown" in record.message for record in caplog.records)

    def test_any_other_reason_is_a_warning_naming_it(self, monkeypatch, caplog, smpmc_tiny):
        self.give_up_with(monkeypatch, "incomplete quantifiers")
        with caplog.at_level(logging.INFO):
            result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task).run()
        assert result.success is False
        assert [record.message for record in caplog.records if "returned unknown" in record.message] == ["SMPMC returned unknown: incomplete quantifiers"]
        assert not any("time limit reached" in record.message for record in caplog.records)

    def test_the_best_assignment_found_so_far_stands(self, monkeypatch, smpmc_tiny_optimality):
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
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_optimality.colored_mdp, smpmc_tiny_optimality.task).run()
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
        """SMPMC on the decision trees of depth 2 of tests/dt-maze (Rmin): the optimum, 20.948..., takes thousands of theory calls to find and prove, a first
        tree only a couple."""
        factory, task = load_sketch("tests/dt-maze")
        return paynt.synthesizer.smpmc.SynthesizerSMPMC(factory.reset_tree(2, general=True), task)

    def test_the_limit_ends_a_single_check_that_would_run_for_much_longer(self, caplog):
        """Seeded just below the optimum, nothing is left to find and the whole search is one solver.check() ending in unsat, which takes thousands of theory
        calls without the limit."""
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

    def test_a_search_stopped_by_the_limit_leaves_the_next_one_alone(self, smpmc_tiny):
        """A Z3_interrupt sent while no check() is running would leave the context cancelled, so that the next solver with a propagator answered sat at once,
        with an empty model."""
        self.maze_synthesizer().synthesize(optimum_threshold=20.9, timeout=0.3)
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny.colored_mdp, smpmc_tiny.task).run()
        assert result.success is True
        prop = smpmc_tiny.task.specification.constraints[0]
        assert smpmc_tiny.colored_mdp.build_assignment(result.assignment).model_check_property(prop).sat

    def test_the_best_assignment_found_before_the_limit_stands(self):
        synthesizer = self.maze_synthesizer()
        synthesizer.synthesize(keep_optimum=True, timeout=1)
        assert synthesizer.best_assignment is not None
        # Rmin: a tree worse than the optimum, since the search was cut short
        assert synthesizer.best_assignment_value > 20.95

    def test_a_robust_search_is_stopped_by_the_limit_too(self, mdp_family_rocks_4_2):
        """There Z3 also runs the theory in quantifier-instantiation sub-contexts."""
        mdp_family_rocks_4_2.task.constraint_name = "exists_forall"
        mdp_family_rocks_4_2.task.forall_pattern = None
        mdp_family_rocks_4_2.task.verify_robust = False
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(mdp_family_rocks_4_2.colored_mdp, mdp_family_rocks_4_2.task)
        started = time.perf_counter()
        synthesizer.synthesize(timeout=0.5)
        assert time.perf_counter() - started < 5
