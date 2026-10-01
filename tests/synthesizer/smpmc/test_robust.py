"""Integration tests of robust synthesis (--constraint exists_forall), each result checked independently with
verify_robust -- AR on the negated specification, a port of molehill's tests/test_robust.py.

smpmc-tiny (see tests/parameter_space/constraints/test_costs.py) has exactly 3 satisfying assignments:
(x1=1,x2=4), (x1=4,x2=3) and (x1=4,x2=4), with x1 in {1..4} and x2 in {3,4}. So "exists x1 forall x2" holds with
x1=4, while "exists x2 forall x1" does not hold at all.
"""

from __future__ import annotations

import gc

import pytest

import paynt.parser.sketch
import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc._utils
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.smpmc.theory

from helpers.helper import get_sketch_paths


def _robust(task, forall_pattern=None, verify_robust=False):
    task.constraint_name = "exists_forall"
    task.forall_pattern = forall_pattern
    task.verify_robust = verify_robust
    return task


def _run(project, props_name, **robust):
    sketch_path, props_path = get_sketch_paths(project, props_name=props_name)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(factory.build(), _robust(task, **robust))
    return synthesizer, task, synthesizer.run()


def _policy(synthesizer, robust_assignment):
    return [
        robust_assignment.parameter_to_option_labels[parameter][robust_assignment.parameter_options(parameter)[0]]
        for parameter in range(robust_assignment.num_parameters)
        if parameter not in synthesizer.forall_parameters
    ]


class TestRobustSynthesisOnFamily:
    """The environment is the family's own parameters, the policy is synthesized."""

    def test_finds_a_policy_robust_against_every_environment(self, rocks_colored_mdp, rocks_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task)).run()
        assert result.success
        robust = result.robust_assignment
        assert [robust.parameter_num_options(parameter) for parameter in range(4)] == [3, 3, 3, 3]
        assert all(robust.parameter_num_options(parameter) == 1 for parameter in range(4, robust.num_parameters))
        assert result.assignment.size == 1

    def test_the_robust_policy_is_confirmed_by_ar(self, rocks_colored_mdp, rocks_task):
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task))
        result = synthesizer.run()
        assert paynt.synthesizer.smpmc._utils.verify_robust(synthesizer.colored_mdp, rocks_task, result.robust_assignment)

    def test_run_reports_the_verification_when_asked(self, rocks_colored_mdp, rocks_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task, verify_robust=True)).run()
        assert result.robust_verified is True

    def test_a_forall_pattern_overrides_the_environment_default(self, rocks_colored_mdp, rocks_task):
        # quantifying the policy universally too: not every policy visits both rocks
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(rocks_colored_mdp, _robust(rocks_task, forall_pattern=".")).run()
        assert not result.success
        assert result.robust_assignment is None


class TestRobustSynthesisOnGenericSketch:
    def test_sat_when_one_option_works_for_every_environment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="^x2$", verify_robust=True)).run()
        assert result.success
        assert str(result.robust_assignment) == "x1=4, x2: {3,4}, x3=8"
        assert result.robust_verified is True

    def test_unsat_when_no_option_works_for_every_environment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="^x1$")).run()
        assert not result.success
        assert result.robust_assignment is None

    def test_a_forall_pattern_is_required_outside_a_family(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        with pytest.raises(ValueError, match="--smpmc-forall"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task))

    def test_a_pattern_matching_no_parameter_is_an_error(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        with pytest.raises(ValueError, match="matches no parameter"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="no_such_hole"))

    def test_an_optimality_objective_optimizes_the_worst_case(self):
        _synthesizer, _task, result = _run("tests/smpmc-tiny", "optimality.props", forall_pattern="^x2$", verify_robust=True)
        assert result.value == pytest.approx(1.0)
        assert str(result.robust_assignment) == "x1=4, x2: {3,4}, x3=8"
        assert result.robust_verified is True

    def test_a_plain_search_reports_no_robust_assignment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, smpmc_tiny_task).run()
        assert result.success
        assert result.robust_assignment is None
        assert result.robust_verified is None


class TestRobustOptimality:
    """The worst case over the environments, optimized.

    On mdp-family-correlated (see its sketch), the best worst case is 0.45 when maximizing, with the policy (a, c), and 0.36 when minimizing, with (b, d).
    """

    def test_maximizes_the_worst_case(self):
        synthesizer, _task, result = _run("tests/mdp-family-correlated", "sketch.props", verify_robust=True)
        assert result.value == pytest.approx(0.45)
        assert _policy(synthesizer, result.robust_assignment) == ["a", "c"]
        assert result.robust_assignment.parameter_num_options(0) == 2, "the environment should be left open"
        assert result.robust_verified is True

    def test_minimizes_the_worst_case(self):
        synthesizer, _task, result = _run("tests/mdp-family-correlated", "min.props", verify_robust=True)
        assert result.value == pytest.approx(0.36)
        assert _policy(synthesizer, result.robust_assignment) == ["b", "d"]
        assert result.robust_verified is True

    def test_over_many_rounds(self):
        """On rocks-4-2 the worst case improves over several rounds before reaching 1.0, as P>=1 is robustly sat (see TestRobustSynthesisOnFamily)."""
        _synthesizer, _task, result = _run("tests/mdp-family-rocks-4-2", "max.props", verify_robust=True)
        assert result.value == pytest.approx(1.0)
        assert result.robust_verified is True

    def test_verification_rejects_a_worst_case_the_policy_does_not_attain(self):
        synthesizer, task, result = _run("tests/mdp-family-correlated", "sketch.props")
        verify = paynt.synthesizer.smpmc._utils.verify_robust
        assert verify(synthesizer.colored_mdp, task, result.robust_assignment, 0.45)
        assert not verify(synthesizer.colored_mdp, task, result.robust_assignment, 0.46)

    def test_the_worst_case_comes_from_the_search_not_from_ar(self, monkeypatch):
        """Each witness's worst case is read off the refutations the search made anyway, see SynthesizerSMPMC._worst_case_of."""

        def fail(*args):
            raise AssertionError("AR was used to evaluate a witness")

        monkeypatch.setattr(paynt.synthesizer.smpmc._utils, "worst_case_by_ar", fail)
        _synthesizer, _task, result = _run("tests/mdp-family-correlated", "min.props")
        assert result.value == pytest.approx(0.36)

    def test_the_ar_fallback_finds_the_same_optimum(self, monkeypatch):
        monkeypatch.setattr(paynt.synthesizer.smpmc.checker.ColoredMdpTheory, "worst_case_bound", lambda self, policy: None)
        _synthesizer, _task, result = _run("tests/mdp-family-correlated", "min.props")
        assert result.value == pytest.approx(0.36)


class TestVerifyRobust:
    """verify_robust on hand-picked candidates: x1 fixed, x2 left open."""

    def _candidate(self, colored_mdp, x1_option):
        return colored_mdp.parameter_space.assume_options_copy([[x1_option], [0, 1], [0]])

    def test_accepts_a_robust_candidate(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        candidate = self._candidate(smpmc_tiny_colored_mdp, x1_option=3)  # x1=4
        assert paynt.synthesizer.smpmc._utils.verify_robust(smpmc_tiny_colored_mdp, smpmc_tiny_task, candidate)

    def test_rejects_a_candidate_violated_by_one_environment(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        candidate = self._candidate(smpmc_tiny_colored_mdp, x1_option=0)  # x1=1: fails for x2=3
        assert not paynt.synthesizer.smpmc._utils.verify_robust(smpmc_tiny_colored_mdp, smpmc_tiny_task, candidate)


class TestFreshPropagators:
    def test_every_term_is_held_through_the_solver_context(self, smpmc_tiny_colored_mdp, smpmc_tiny_task):
        """Z3 frees an MBQI sub-context once its round ends, while z3's Python API keeps every propagator alive until
        exit: a term held through the sub-context is then released through a dangling pointer. The resulting heap
        corruption only shows up at exit, and not reliably (see test_synthesizer_smpmc_cli.py), so check the invariant."""
        paynt.synthesizer.smpmc.SynthesizerSMPMC(smpmc_tiny_colored_mdp, _robust(smpmc_tiny_task, forall_pattern="^x2$")).run()
        propagators = [obj for obj in gc.get_objects() if isinstance(obj, paynt.synthesizer.smpmc.theory.SmpmcPropagator)]
        assert any(propagator.fresh_ctx is not None for propagator in propagators), "expected MBQI to create sub-context propagators"
        for propagator in propagators:
            assert propagator.owner_ctx is not propagator.fresh_ctx
            assert all(term.ctx is propagator.owner_ctx for term in propagator.term_of.values())
