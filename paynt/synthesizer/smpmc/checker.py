"""Algorithm 1 "CheckMDPs" (arXiv:2511.08078): decide whether a `viable(...)`/`not viable(...)` literal over a partial parameter assignment is refuted, by
model-checking the induced sub-MDP with Storm.

Ported from molehill's Mole.partial_model_consistent + counterexamples.check
(https://github.com/linusheck/molehill, GPL-3.0), but built directly on PAYNT's own primitives instead
of molehill's bespoke fastmole/MatrixGenerator and modelchecker.py:
  - the induced sub-MDP C[eta] is ColoredMdp.build(eta) (paynt/colored_mdp.py), not a separate C++
    extension -- PAYNT's own coloring.selectCompatibleChoices already does this;
  - V^max/V^min of C[eta] is SubMdp.model_check_property(prop, alt=...) (paynt/model/model.py), which
    already gives both optimization directions via prop.formula/prop.formula_alt -- the same mechanism
    AR already uses for its own primary/secondary bounds;
  - the conflict-minimization step (paper's Theorem 6: drop parameters not reachable in C[eta]) uses
    coloring.getStateToHoles() unioned over the induced sub-MDP's reachable states, rather than porting
    molehill's own (partly dead-code) binary-search minimizer. For a general coloring those static per-state
    supports are useless (in a decision tree every state mentions every parameter), so it asks the coloring
    which parameters actually excluded a choice at the reachable states instead: ColoringGeneral.relevantParameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import payntbind.synthesis
import stormpy

import paynt.colored_mdp
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.specification.property_result
import paynt.synthesizer.smpmc.cache
import paynt.synthesizer.statistic
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)

#: relative-error threshold above which a value-iteration result is treated as non-converged rather than
#: as ordinary floating-point disagreement between two different (but both legitimate) numerical methods.
#: The failure mode this guards against is not subtle -- a capped, non-converged VI result differs from
#: the true (policy-iteration-computed) value by orders of magnitude, not by a few ULPs -- so this can
#: afford to be generous without risking false positives.
_REWARD_CONVERGENCE_RELATIVE_TOLERANCE = 0.01

#: sub-MDP state-count above which a policy-iteration verification call is skipped entirely, rather than
#: attempted with an iteration cap. Confirmed empirically that capping iterations (both the outer minmax
#: loop and the inner linear-equation solve each outer iteration depends on) does not help here: on a
#: 1,652,566-state model, even a single outer iteration capped to just 10 inner iterations took over 40
#: seconds, and every larger cap tried was worse -- the cost is dominated by the sheer size of the sparse
#: system a single call has to set up and touch, not by how many iterations run. Below this threshold,
#: verification is cheap and valuable (the feature's own motivating case, generic-maze, is 183 states);
#: at or above it, an unconditional per-direction PI cross-check would make SMPMC impractical on
#: realistically large models, for a safety net AR/CEGIS/Hybrid already go without everywhere. Calibrated
#: from only two real data points (183 states: fine; 1.65M states: catastrophic), so treat this as a
#: coarse heuristic, not a precisely-tuned value -- revisit if a real model lands in between and disagrees
#: with it either way.
_PI_VERIFICATION_MAX_STATES = 50_000


@dataclass
class TheoryResult:
    """A refutation: the Theorem-6-minimized set of parameters responsible, and the value that triggered it."""

    conflict_parameters: list[int]
    value: Any


@dataclass
class WorstCase:
    """A bound on a policy's worst case over the environments (at least as bad as that worst case, up to model checking precision); exact if some environment
    attains it."""

    value: Any
    exact: bool


class ColoredMdpTheory:
    """Owns the Storm-calling half of the theory solver.

    Stateless across polarities/literals except for the epoch counter (bumped once per optimum update, see the optimality loop in synthesizer.py), the result
    cache and the worst-case bounds -- one instance is shared by every SmpmcPropagator produced via Z3's fresh() during MBQI, so all of them stay consistent
    across quantifier-instantiation sub-contexts.
    """

    def __init__(
        self,
        colored_mdp: paynt.colored_mdp.ColoredMdp,
        prop: paynt.specification.property.Property,
        stat: paynt.synthesizer.statistic.Statistic | None = None,
        forall_parameters: list[int] | None = None,
    ):
        """:param forall_parameters: the universally quantified parameters of a robust search; the remaining ones, the policy, key worst_case_bound()"""
        self.colored_mdp = colored_mdp
        self.prop = prop
        # optional: not needed by the propagator tests (test_propagator.py -- a fake theory that never
        # touches this class at all) or the pure checker.check() tests (test_theory.py -- real Storm calls
        # but no interest in progress reporting), only by a real synthesis run (synthesizer.py always
        # passes one). When present, every model check feeds paynt.synthesizer.statistic.Statistic's
        # existing DTMC/MDP iteration counters and throttled status-line logging (the same mechanism
        # AR/CEGIS already use via self.stat.iteration(...)), and every fresh refutation accumulates an
        # "explored" estimate onto the synthesizer, mirroring SmtSolver.exclude_conflict's
        # pruning_estimate for CEGIS -- see check()'s use of it below.
        self.stat = stat
        # kappa restricted to "which parameters are relevant at this state", unioned over an induced
        # sub-MDP's reachable states to minimize a conflict (Theorem 6) -- one BitVector of parameter
        # indices per underlying-MDP state. Only for standard colorings: see relevant_parameters().
        self.state_to_parameters: list[Any] | None = None if colored_mdp.has_general_coloring else colored_mdp.coloring.getStateToHoles()
        self.cache = paynt.synthesizer.smpmc.cache.PartialModelCache()
        # bumped by the optimality loop (synthesizer.py) each time the threshold tightens, so cached
        # verdicts computed under a looser threshold that may no longer hold aren't wrongly reused -- see
        # PartialModelCache's docstring
        self.epoch = 0
        self.mc_calls = 0
        # an error the theory raised inside a Z3 callback, which the propagator cannot raise itself: the synthesizer raises it once check() returns
        self.failure: Exception | None = None
        # whether the time limit is reached, set by the synthesizer: the propagator asks at every callback and interrupts the search once it is (never anywhere
        # else, see SynthesizerSMPMC._limit_solver_time)
        self.time_is_up: Callable[[], bool] = lambda: False

        # Robust search: Z3 only reports a policy once every environment has been covered by refuted `not viable` literals, each of which comes with its
        # model-checked worst case over its region -- so the worst of those, per policy, bounds the policy's worst case over all environments, and beats the
        # current threshold. See worst_case_bound(); reset with every epoch, as the covering is only valid under the threshold it was refuted against.
        self.policy_parameters: list[int] | None = None
        if forall_parameters:
            forall = set(forall_parameters)
            self.policy_parameters = [parameter for parameter in range(colored_mdp.parameter_space.num_parameters) if parameter not in forall]
        self._worst_case: dict[tuple[int, ...], WorstCase] = {}
        # policies with a refutation of unknown value: no bound
        self._worst_case_unknown: set[tuple[int, ...]] = set()
        self._worst_case_epoch = 0

        # Set once the first fully-fixed parameter assignment has been checked (via check()), regardless
        # of its verdict -- see theory.py's _analyse(), which skips every *partial* query until this is
        # True, mirroring molehill's own "check a DTMC first" heuristic. Lives here (not on the
        # propagator) because it's a property of the search as a whole, shared across every
        # SmpmcPropagator instance fresh() creates during MBQI, exactly like epoch/cache.
        self.first_full_assignment_checked = False

        # Reward properties only (see _model_check): PAYNT's default model-checking environment
        # (paynt.specification.property.Property.environment) caps value-iteration at a fixed number of
        # iterations to avoid hanging forever (see the plan's Phase 0), but on some models one direction
        # of a reward property converges pathologically slowly -- capped VI then silently returns whatever
        # partial snapshot it reached, which can be wrong by orders of magnitude and is unsound to use for
        # a refutation decision. Verified once per direction (alt=False/True), lazily, against policy
        # iteration (which solves the exact fixed-point system directly, so it isn't affected by the same
        # slow-convergence failure mode as VI -- see _pi_environment for its own, separate iteration cap);
        # whichever method actually works for that direction is then used for the rest of this
        # ColoredMdpTheory's lifetime, so the comparison cost is paid at most twice per run, not once per
        # query. Scoped locally to SMPMC rather than changing paynt.specification.property's shared
        # default, since AR/CEGIS/Hybrid already tolerate this (their bounds only affect pruning, and any
        # answer they report gets independently re-verified as a concrete DTMC before being trusted) in a
        # way SMPMC's Z3-learned refutations do not.
        self._reward_pi_environment: Any = None
        self._reward_convergence_verified: dict[bool, bool] = {}
        self._reward_use_policy_iteration: dict[bool, bool] = {}
        # throttles the "skipping policy-iteration verification" log line (see _model_check) to once per
        # direction -- purely a logging concern, unlike _reward_convergence_verified this never suppresses
        # a retry of the actual (cheap) size check on a later, possibly smaller sub-MDP.
        self._reward_verification_skip_warned: dict[bool, bool] = {}

    def check(self, fixed: dict[int, int], polarity: bool) -> TheoryResult | None:
        """:param fixed: {parameter_index: option} for every parameter currently decided in the partial
            model Z3 is exploring.
        :param polarity: True to check the `viable(...)` literal, False for `not viable(...)`.
        :returns: None if inconclusive (this partial assignment could still go either way), else a
            TheoryResult describing the refutation to hand back to Z3 as a learned conflict.
        """
        # The BitVec encoding's width has spare headroom beyond what any parameter's option count needs
        # (see paynt/parameter_space/bitvec.py), so Z3 can transiently fix a variable to a value outside
        # its actual option range while still exploring whether the plain Boolean-level in_range()
        # constraint holds -- that constraint alone will reject such a branch without any help from this
        # theory, so the correct (and, empirically, necessary -- an out-of-range option previously reached
        # ColoredMdp.build() and crashed stormpy's submodel construction with a deadlock-state error) move
        # here is to treat it as inconclusive and let ordinary constraint solving handle it.
        for parameter, option in fixed.items():
            if option not in self.colored_mdp.parameter_space.parameter_options(parameter):
                return None

        cached = self.cache.lookup(fixed, polarity, self.epoch)
        if cached is None:
            return None
        if cached is not paynt.synthesizer.smpmc.cache.MISS:
            refutation = min(cached, key=lambda refutation: len(refutation.conflict_parameters))
            if polarity is False:
                # each cached region contains the query's, so each value bounds it; the tightest does so best
                known = [refutation for refutation in cached if refutation.value is not None]
                exact_refutations = [refutation for refutation in known if refutation.exact]
                if exact_refutations:
                    self._record_worst_case(fixed, exact_refutations[0].value, True)
                elif known:
                    tightest = min if self.prop.minimizing else max
                    self._record_worst_case(fixed, tightest(refutation.value for refutation in known), False)
                else:
                    self._record_worst_case(fixed, None, False)
            return TheoryResult(refutation.conflict_parameters, refutation.value)

        eta = self.colored_mdp.parameter_space.copy()
        for parameter, option in fixed.items():
            eta.parameter_set_options(parameter, [option])

        if eta.size == 1:
            # Every parameter is fixed. ColoredMdp.build() would still hand back an MDP-typed model here
            # (even though it has exactly one choice per state) -- and solving that via minmax/policy
            # iteration can disagree with the exact DTMC computation build_assignment() gives for the very
            # same deterministic process, by an amount just past model_checking_precision (confirmed
            # empirically: two different underlying linear-equation solves of a mathematically identical
            # system, not a convergence failure -- policy iteration is exact either way). That tiny
            # disagreement is enough to make a non-improving witness look viable to the theory when it
            # shouldn't. build_assignment() sidesteps this by computing the exact value directly (and is
            # cheaper besides). It keeps an MDP where the assignment leaves genuine nondeterminism -- an
            # incomplete coloring, or a "family"/"pomdp_family" ColoredMdp (the agent's policy isn't itself a
            # declared parameter there) -- so this shortcut is correct unconditionally, not just for the
            # common case: the value is then that of the best resolution of the choices left, which is what
            # `viable` asks for.
            sub_mdp = self.colored_mdp.build_assignment(eta)
            paynt.utils.error_handling.require_markov_chain_for_robust_search(sub_mdp, self.policy_parameters)
        else:
            # deliberately no parent_selected_choices reuse hint here: that optimization assumes
            # monotonically-nested parameter spaces (true for AR's splitting), which does not hold for
            # Z3's arbitrary CDCL backtracking -- see the plan's design-decisions section.
            sub_mdp, _selected_choices = self.colored_mdp.build(eta)
        self.mc_calls += 1
        if self.stat is not None:
            # classifies DTMC vs MDP by inspecting sub_mdp.model's stormpy type and throttles the actual
            # log line to once every Statistic.status_period_seconds -- exactly what AR/CEGIS get from the
            # same call
            self.stat.iteration(sub_mdp)

        # alt=(polarity is False): prop.formula is already set (via minimizing/optimality_type) to
        # compute the value in *this property's own success direction* -- so alt=False always gives the
        # best-case value for the `viable` check, and alt=True (formula_alt, the opposite direction)
        # always gives the worst-case value for the `not viable` check, regardless of whether prop is a
        # minimizing or maximizing property. See Algorithm 1: (viable -> max, "<"), (not viable -> min, ">=").
        result = self._model_check(sub_mdp, alt=(polarity is False))

        if result.sat is polarity:
            # best case (viable) still meets the threshold, or worst case (not viable) still misses it --
            # this partial assignment does not yet decide the literal either way
            self.cache.insert_inconclusive(fixed, polarity, self.epoch)
            return None

        # refuted: minimize the conflict by dropping every fixed parameter that isn't reachable (hence
        # irrelevant) in the induced sub-MDP
        relevant_parameters = self.relevant_parameters(eta, sub_mdp)
        conflict_parameters = [parameter for parameter in fixed if parameter in relevant_parameters]

        if self.stat is not None:
            # "explored" estimate for this refutation: the number of complete assignments it rules out is
            # the product of every *other* parameter's option count, since conflict_parameters is exactly
            # the (Theorem-6-minimized) set that had to be fixed to reach this refutation -- any value of
            # every parameter *not* in it is still covered. Mirrors SmtSolver.exclude_conflict's own
            # pruning_estimate for CEGIS (paynt/parameter_space/smt.py), including its same caveat: since
            # nothing guarantees learned conflicts are pairwise disjoint, this is an estimate that can
            # double-count overlapping regions, not an exact count -- CEGIS's existing "explored" figure
            # has always carried the identical caveat, so this isn't a new source of imprecision, just the
            # same one on a new engine. Only charged once per fresh refutation (this line is unreachable
            # from the cache-hit path above), never on a cache replay of the same conflict.
            pruning_estimate = math.prod(
                self.colored_mdp.parameter_space.parameter_num_options(parameter)
                for parameter in range(self.colored_mdp.parameter_space.num_parameters)
                if parameter not in conflict_parameters
            )
            assert self.stat.synthesizer.explored is not None
            self.stat.synthesizer.explored += pruning_estimate

        exact = eta.size == 1
        self.cache.insert_refuted(fixed, polarity, conflict_parameters, self.epoch, result.value, exact)
        if polarity is False:
            self._record_worst_case(fixed, result.value, exact)
        return TheoryResult(conflict_parameters, result.value)

    def relevant_parameters(self, eta: paynt.parameter_space.parameter_space.ParameterSpace, sub_mdp: Any) -> set[int]:
        """The parameters the sub-MDP C[eta] depends on (Theorem 6): fixing any other parameter differently cannot change it, so a conflict may drop them."""
        states = sub_mdp.underlying_mdp_state_map
        if self.colored_mdp.has_general_coloring:
            # Ask the coloring which parameters actually excluded a choice at the reachable states under eta. Widening the others cannot add a choice there,
            # so the sub-MDP -- and with it the refutation -- is unchanged (ColoringGeneral.relevantParameters).
            return set(self.colored_mdp.coloring.relevantParameters(eta.native, list(states)))
        assert self.state_to_parameters is not None
        # The union is a chain of BitVector ORs, done in C++: unioning Python sets instead iterates every BitVector from Python, which takes most of the time
        # of a check on a large sub-MDP
        relevant = stormpy.BitVector(self.colored_mdp.parameter_space.num_parameters, False)
        for state in states:
            relevant |= self.state_to_parameters[state]
        return set(relevant)

    def _record_worst_case(self, fixed: dict[int, int], value: Any, exact: bool) -> None:
        """Account a refuted `not viable` literal, whose region's worst case is (bounded by) value, to its policy -- if the policy is fully fixed."""
        if self.policy_parameters is None or any(parameter not in fixed for parameter in self.policy_parameters):
            return
        self._sync_worst_case()
        policy = tuple(fixed[parameter] for parameter in self.policy_parameters)
        if value is None:
            self._worst_case_unknown.add(policy)
            return
        current = self._worst_case.get(policy)
        if current is None or (value > current.value if self.prop.minimizing else value < current.value):
            self._worst_case[policy] = WorstCase(value, exact)
        elif value == current.value and exact:
            current.exact = True

    def _sync_worst_case(self) -> None:
        if self._worst_case_epoch != self.epoch:
            self._worst_case = {}
            self._worst_case_unknown = set()
            self._worst_case_epoch = self.epoch

    def worst_case_bound(self, policy: dict[int, int]) -> WorstCase | None:
        """A bound on the worst case over the environments of policy -- sound once Z3 has reported policy as robust under the current threshold, as it then
        refuted every environment's `not viable` literal; None if there is no such bound.

        :param policy: {parameter: option} for every policy parameter
        """
        assert self.policy_parameters is not None, "worst-case bounds are only kept for a robust search"
        self._sync_worst_case()
        key = tuple(policy[parameter] for parameter in self.policy_parameters)
        if key in self._worst_case_unknown:
            return None
        return self._worst_case.get(key)

    def _model_check(self, sub_mdp: Any, alt: bool) -> paynt.specification.property_result.PropertyResult:
        """Like sub_mdp.model_check_property(self.prop, alt=alt), but for reward properties verifies (once per direction) that PAYNT's default capped value-
        iteration environment actually converged, falling back to policy iteration for this direction's remaining queries if it didn't.

        See __init__.
        """
        if not self.prop.reward:
            return sub_mdp.model_check_property(self.prop, alt=alt)

        if self._reward_use_policy_iteration.get(alt, False):
            return self._model_check_with_environment(sub_mdp, alt, self._pi_environment())

        vi_result = sub_mdp.model_check_property(self.prop, alt=alt)
        if self._reward_convergence_verified.get(alt, False):
            return vi_result

        if sub_mdp.model.nr_states >= _PI_VERIFICATION_MAX_STATES:
            # See _PI_VERIFICATION_MAX_STATES: a single PI call is not affordable on a sub-MDP this large,
            # regardless of iteration caps, so there is nothing to verify against here -- trust VI for
            # *this* query, but deliberately do NOT set _reward_convergence_verified[alt]: a later query
            # for this same direction can land on a much smaller sub-MDP (a full assignment is exactly
            # such a case, and skips straight to build_assignment()'s DTMC), where verification is cheap
            # again and still worth actually doing. Marking this direction "verified" here (as an earlier
            # version of this method did) would have permanently disabled the safety net for the rest of
            # the run the moment the very first query happened to be large -- confirmed as a real bug: it
            # let a catastrophically wrong VI value on a full assignment (a DTMC, ~10^180 vs the true
            # ~10^-13) through unquestioned, because "verified" had already been latched on an earlier,
            # skipped, much larger query.
            if not self._reward_verification_skip_warned.get(alt, False):
                self._reward_verification_skip_warned[alt] = True
                logger.warning(
                    f"skipping policy-iteration verification for {'the alt' if alt else 'the primary'} "
                    f"direction of {self.prop}: the induced sub-MDP has {sub_mdp.model.nr_states} states, "
                    "too large to verify affordably -- trusting value iteration's own result until a "
                    "smaller sub-MDP makes verification affordable (this warning will not repeat)"
                )
            return vi_result

        self._reward_convergence_verified[alt] = True
        pi_result = self._model_check_with_environment(sub_mdp, alt, self._pi_environment())
        vi_value, pi_value = vi_result.value, pi_result.value
        relative_error = abs(vi_value - pi_value) / max(abs(pi_value), 1e-12)
        if relative_error <= _REWARD_CONVERGENCE_RELATIVE_TOLERANCE:
            return vi_result

        logger.warning(
            f"value iteration did not converge for {'the alt' if alt else 'the primary'} direction of "
            f"{self.prop}: got {vi_value}, policy iteration gives {pi_value}; using policy iteration for "
            "the rest of this run"
        )
        self._reward_use_policy_iteration[alt] = True
        return pi_result

    def _model_check_with_environment(self, sub_mdp: Any, alt: bool, environment: Any) -> paynt.specification.property_result.PropertyResult:
        formula = self.prop.formula if not alt else self.prop.formula_alt
        result = stormpy.model_checking(sub_mdp.model, formula, extract_scheduler=True, environment=environment)
        value = result.at(sub_mdp.initial_state)
        return paynt.specification.property_result.PropertyResult(self.prop, result, value)

    def _pi_environment(self) -> Any:
        if self._reward_pi_environment is None:
            env = stormpy.Environment()
            env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.policy_iteration
            # Policy iteration solves the exact fixed-point system directly, so in principle it shouldn't
            # need an iteration cap at all -- but on some MDP structures (confirmed on a 165-state
            # POMDP-derived model) Storm's PI solver doesn't satisfy its own convergence check even after
            # 10000 iterations, so cap it too rather than trust it to always self-terminate. Empirically
            # (5 consecutive trials on that same model) the *value* it reports is identical to 7 significant
            # figures at this cap and takes well under a second, matching VI's own capped value on the same
            # model -- so a capped-but-technically-"non-converged" PI result is still trustworthy here.
            if not self.prop.sound:
                payntbind.synthesis.set_max_iterations_minmax(env.solver_environment.minmax_solver_environment, self.prop.max_minmax_iterations)
            self._reward_pi_environment = env
        return self._reward_pi_environment
