"""Pure unit tests of SmpmcPropagator against a fake theory -- no Storm, no real colored MDP.

This is deliberately the cheapest and most direct test of the Z3 UserPropagateBase integration itself, isolated from whether Storm/model-checking is doing the
right thing (see test_theory.py for that).

The FakeTheory below models `viable(p0, p1, p2)` as "p0 + p1 + p2 >= 3" -- simple enough to reason about by hand, but with three parameters (not two)
specifically because the bug this test suite is built to catch (see theory.py's module docstring) only shows up once at least one parameter is not a live Z3
variable at `created` time. That happens for the real engine's `--forall`-free single-option holes and, here, whenever a domain constraint pins a parameter to
one value before the propagator inspects the viable(...) term -- both are exercised below.
"""

from __future__ import annotations

import z3

import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.smpmc.theory


class FakeTheory:
    """Viable(p0, p1, p2) iff p0 + p1 + p2 >= 3."""

    def __init__(self):
        self.calls: list[tuple[dict, bool]] = []
        self.first_full_assignment_checked = False
        # the parts of ColoredMdpTheory's state SmpmcPropagator watches (see SmpmcPropagator._theory_state), and whether the time limit is reached
        self.mc_calls = 0
        self.epoch = 0
        self.time_is_up = lambda: False

    def check(self, fixed: dict, polarity: bool):
        self.calls.append((dict(fixed), polarity))
        if len(fixed) < 3:
            return None
        value = sum(fixed.values())
        actual_viable = value >= 3
        if actual_viable is polarity:
            return None
        return paynt.synthesizer.smpmc.checker.TheoryResult(conflict_parameters=sorted(fixed.keys()), value=value)


def _build_solver(width, p0_max, p1_max, p2_max, force_p2_to_zero=True):
    solver = z3.Solver()
    p0 = z3.BitVec("p0", width)
    p1 = z3.BitVec("p1", width)
    p2 = z3.BitVec("p2", width)
    name_to_parameter = {"p0": 0, "p1": 1, "p2": 2}
    viable = z3.PropagateFunction("viable", z3.BitVecSort(width), z3.BitVecSort(width), z3.BitVecSort(width), z3.BoolSort())
    theory = FakeTheory()
    _propagator = paynt.synthesizer.smpmc.theory.SmpmcPropagator(solver, None, theory, name_to_parameter)

    solver.add(z3.ULE(p0, p0_max))
    solver.add(z3.ULE(p1, p1_max))
    if force_p2_to_zero:
        # this is what makes p2 constant-folded away by Z3's preprocessor before `created` ever sees it as
        # a live variable -- the scenario the positional const/var tracking in theory.py exists for
        solver.add(p2 == 0)
    else:
        solver.add(z3.ULE(p2, p2_max))
    solver.add(viable(p0, p1, p2))
    return solver, (p0, p1, p2), theory


class TestExistentialSearch:
    def test_finds_a_genuinely_viable_witness(self):
        solver, (p0, p1, p2), theory = _build_solver(width=3, p0_max=3, p1_max=1, p2_max=0)
        result = solver.check()
        assert result == z3.sat
        model = solver.model()
        v0, v1, v2 = model[p0].as_long(), model[p1].as_long(), model[p2].as_long()
        assert v0 + v1 + v2 >= 3, f"propagator accepted a non-viable witness: ({v0},{v1},{v2})"

    def test_refutes_when_no_witness_can_be_viable(self):
        # max achievable sum is 1+0+0=1, never >= 3
        solver, _vars, theory = _build_solver(width=3, p0_max=1, p1_max=0, p2_max=0)
        assert solver.check() == z3.unsat
        assert len(theory.calls) > 0, "expected the theory to actually be consulted before concluding unsat"

    def test_constant_folded_parameter_still_participates_in_theory_checks(self):
        """Regression test for the bug this module's docstring describes: p2 is pinned to 0 via a plain equality, so Z3 constant-folds it into viable(p0, p1, 0)
        before `created` fires.

        A propagator that only tracks live-variable arguments would build `fixed` dicts missing parameter 2 entirely, so FakeTheory.check (which requires
        len(fixed) == 3) would never actually run -- silently returning `sat` for a non-viable witness. This asserts both the specific failure mode and its
        absence.
        """
        solver, (p0, p1, p2), theory = _build_solver(width=3, p0_max=1, p1_max=0, p2_max=0, force_p2_to_zero=True)
        result = solver.check()
        assert result == z3.unsat
        # the crucial assertion: some theory.check call actually received all three parameters, including
        # the constant-folded one
        assert any(len(fixed) == 3 and 2 in fixed for fixed, _polarity in theory.calls), (
            f"parameter 2 (constant-folded to 0) never reached the theory; calls were: {theory.calls}"
        )

    def test_repeated_trials_are_not_flaky(self):
        bad = 0
        for _ in range(20):
            solver, (p0, p1, p2), _theory = _build_solver(width=3, p0_max=3, p1_max=1, p2_max=0)
            result = solver.check()
            if result != z3.sat:
                bad += 1
                continue
            model = solver.model()
            if model[p0].as_long() + model[p1].as_long() + model[p2].as_long() < 3:
                bad += 1
        assert bad == 0


class TestTimeIsUp:
    """Once the theory says the time limit is reached, the propagator stops the running search at its next callback, and does not consult the theory any more.

    (SmpmcPropagator._interrupt_if_time_is_up)
    """

    def test_the_search_stops_without_consulting_the_theory(self):
        solver, _vars, theory = _build_solver(width=3, p0_max=1, p1_max=1, p2_max=0)
        theory.time_is_up = lambda: True
        result = solver.check()
        # never a refutation (the search was cut off), and a sat here would be an unchecked candidate: the synthesizer discards what comes back past the limit
        assert result != z3.unsat
        assert theory.calls == []

    def test_the_search_stops_when_the_time_runs_out_in_the_middle_of_it(self):
        solver, _vars, theory = _build_solver(width=3, p0_max=1, p1_max=1, p2_max=0)
        theory.time_is_up = lambda: len(theory.calls) >= 2
        solver.check()
        assert len(theory.calls) == 2

    def test_a_search_stopped_that_way_leaves_the_next_one_alone(self):
        """An interrupt sent while no check() runs would leave the context cancelled, and the next solver with a propagator would answer sat at once."""
        solver, _vars, theory = _build_solver(width=3, p0_max=1, p1_max=1, p2_max=0)
        theory.time_is_up = lambda: True
        solver.check()
        solver, (p0, p1, p2), theory = _build_solver(width=3, p0_max=3, p1_max=1, p2_max=0)
        assert solver.check() == z3.sat
        model = solver.model()
        assert model[p0].as_long() + model[p1].as_long() + model[p2].as_long() >= 3
        assert len(theory.calls) > 0


class TestChecksAFullAssignmentFirst:
    """Regression coverage for the "check a full assignment first" heuristic ported from molehill's own Mole.partial_model_consistent (its "magic trick"): a
    genuinely partial query (some but not all parameters fixed) must never reach the theory before at least one fully-fixed one has.

    Found necessary on a real 1.65M-state model where a threshold so generous that no partial query is ever refutable turned this into pure overhead -- an
    expensive model check paid once per parameter Z3 fixes on the way to its first candidate, for no pruning benefit. See theory.py's module docstring.
    """

    def test_no_partial_query_reaches_the_theory_before_the_first_full_one(self):
        solver, _vars, theory = _build_solver(width=3, p0_max=3, p1_max=1, p2_max=0)
        solver.check()
        first_full_index = next(i for i, (fixed, _polarity) in enumerate(theory.calls) if len(fixed) == 3)
        for fixed, _polarity in theory.calls[:first_full_index]:
            assert len(fixed) in (0, 3), (
                f"a genuinely partial query ({fixed}) reached the theory before any full assignment did: {theory.calls[: first_full_index + 1]}"
            )

    def test_partial_queries_resume_after_the_first_full_one(self):
        # p0 and p1 both genuinely live (p1_max=1, not 0 -- a single-valued range would be constant-folded
        # away like p2 is, leaving no room for a partial state at all); max achievable sum is 1+1+0=2 < 3,
        # so every one of the 4 combinations is refuted, forcing Z3 to keep backtracking and re-deciding
        # both variables, giving partial queries repeated opportunities to run once
        # first_full_assignment_checked flips to True
        solver, _vars, theory = _build_solver(width=3, p0_max=1, p1_max=1, p2_max=0)
        solver.check()
        first_full_index = next(i for i, (fixed, _polarity) in enumerate(theory.calls) if len(fixed) == 3)
        later_calls = theory.calls[first_full_index + 1 :]
        assert any(0 < len(fixed) < 3 for fixed, _polarity in later_calls), (
            "expected at least one genuinely partial query to reach the theory after the first full "
            f"assignment was checked; calls after that point: {later_calls}"
        )

    def test_the_flag_is_shared_across_fresh_propagator_instances(self):
        """Fresh() propagators (MBQI sub-contexts) share theory, so a full assignment checked via one instance must be visible to another -- this is what lets
        partial-query pruning resume globally, not per sub-context."""
        theory = FakeTheory()
        propagator_a = paynt.synthesizer.smpmc.theory.SmpmcPropagator(z3.Solver(), None, theory, {})
        propagator_b = propagator_a.fresh(None)
        assert propagator_b.theory is theory
        theory.first_full_assignment_checked = True
        assert propagator_a.theory.first_full_assignment_checked is True


class TestPushSkipsRedundantAnalyses:
    """A push re-analyses only if a value was fixed or the (shared) theory changed since the enclosing push; driven by hand, outside any Z3 search, on a
    viable(1, 1, 1) literal fixed to true -- viable for FakeTheory, so no conflict is ever pushed."""

    def _propagator(self):
        theory = FakeTheory()
        propagator = paynt.synthesizer.smpmc.theory.SmpmcPropagator(z3.Solver(), None, theory, {})
        propagator.viable_args["v"] = [("const", 1), ("const", 1), ("const", 1)]
        propagator.partial_model["v"] = True
        propagator.trail.append("v")
        return propagator, theory

    def test_an_unchanged_push_skips_the_analysis(self):
        propagator, theory = self._propagator()
        propagator.push()
        propagator.push()
        assert len(theory.calls) == 1

    def test_a_newly_fixed_value_is_analysed(self):
        propagator, theory = self._propagator()
        propagator.push()
        propagator.partial_model["x"] = 0
        propagator.trail.append("x")
        propagator.push()
        assert len(theory.calls) == 2

    def test_a_theory_change_is_analysed(self):
        """e.g. a fresh() propagator sharing the theory made a Storm call in between, whose cached verdict may now refute a query here."""
        propagator, theory = self._propagator()
        propagator.push()
        theory.mc_calls += 1
        propagator.push()
        assert len(theory.calls) == 2

    def test_pop_drops_the_scope_state_with_the_scope(self):
        propagator, _theory = self._propagator()
        propagator.push()
        propagator.push()
        propagator.pop(1)
        assert len(propagator.scopes) == len(propagator.scope_states) == 1


class RobustFakeTheory:
    """Viable(x0, x1, y0) iff x0 + x1 >= y0 -- a different predicate from FakeTheory above (deliberately:

    this is what exercises the forall-quantified/MBQI path meaningfully, since it depends on the third argument, whereas the plain-existential tests above never
    vary the "environment" argument).
    """

    def __init__(self):
        self.calls: list[tuple[dict, bool]] = []
        self.first_full_assignment_checked = False
        # the parts of ColoredMdpTheory's state SmpmcPropagator watches (see SmpmcPropagator._theory_state), and whether the time limit is reached
        self.mc_calls = 0
        self.epoch = 0
        self.time_is_up = lambda: False

    def check(self, fixed: dict, polarity: bool):
        self.calls.append((dict(fixed), polarity))
        if len(fixed) < 3:
            return None
        value = fixed[0] + fixed[1] - fixed[2]
        actual_viable = value >= 0
        if actual_viable is polarity:
            return None
        return paynt.synthesizer.smpmc.checker.TheoryResult(conflict_parameters=sorted(fixed.keys()), value=value)


class TestRobustSearch:
    """The forall-quantified (MBQI) path: exists x0,x1 .

    forall y0 in [0,y_max] . viable(x0,x1,y0).
    """

    def _build_robust_solver(self, width, x_max, y_max):
        solver = z3.Solver()
        x0 = z3.BitVec("x0", width)
        x1 = z3.BitVec("x1", width)
        y0 = z3.BitVec("y0", width)
        name_to_parameter = {"x0": 0, "x1": 1, "y0": 2}
        viable = z3.PropagateFunction("viable", z3.BitVecSort(width), z3.BitVecSort(width), z3.BitVecSort(width), z3.BoolSort())
        theory = RobustFakeTheory()
        propagator = paynt.synthesizer.smpmc.theory.SmpmcPropagator(solver, None, theory, name_to_parameter)
        solver.add(z3.ULE(x0, x_max))
        solver.add(z3.ULE(x1, x_max))
        solver.add(z3.ForAll([y0], z3.Implies(z3.ULE(y0, y_max), viable(x0, x1, y0))))
        return solver, (x0, x1, y0), theory, propagator

    def test_finds_a_witness_robust_against_every_environment(self):
        # max(x0+x1) = 3+3 = 6 >= y_max = 4: some witness must dominate every y0 in [0,4]
        solver, (x0, x1, _y0), _theory, _propagator = self._build_robust_solver(width=4, x_max=3, y_max=4)
        assert solver.check() == z3.sat
        model = solver.model()
        vx0 = model.eval(x0, model_completion=True).as_long()
        vx1 = model.eval(x1, model_completion=True).as_long()
        assert vx0 + vx1 >= 4, f"witness ({vx0},{vx1}) is not actually robust against y_max=4"

    def test_refutes_when_no_witness_can_be_robust(self):
        # max(x0+x1) = 3+3 = 6 < y_max = 10: no witness can dominate every y0 in [0,10]
        solver, _vars, _theory, _propagator = self._build_robust_solver(width=4, x_max=3, y_max=10)
        assert solver.check() == z3.unsat
