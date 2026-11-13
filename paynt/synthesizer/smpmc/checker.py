"""Algorithm 1 "CheckMDPs" (arXiv:2511.08078): decide whether a `viable(...)`/`not viable(...)` literal over a partial parameter assignment is refuted, by
model-checking the induced sub-MDP with Storm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

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

#: relative error between the value-iteration and the policy-iteration value above which value iteration counts as not converged. A capped, non-converged result
#: is off by orders of magnitude, so the threshold can be generous.
# TODO once Storm has a new release we should be able to check for convergence
_REWARD_CONVERGENCE_RELATIVE_TOLERANCE = 0.01

#: sub-MDP size (in states) from which the policy-iteration cross-check is skipped: one call is too expensive on a very large sub-MDP whatever the iteration
#: cap, as the cost is in setting up the sparse system. A coarse heuristic.
_PI_VERIFICATION_MAX_STATES = 50_000


@dataclass
class TheoryResult:
    """A refutation: the minimized set of parameters responsible, and the value that triggered it."""

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
        self.stat = stat
        # parameters per underlying-MDP state (a BitVector each), to minimize a conflict; only for standard colorings, see relevant_parameters()
        self.state_to_parameters: list[Any] | None = None if colored_mdp.has_general_coloring else colored_mdp.coloring.getStateToHoles()
        self.cache = paynt.synthesizer.smpmc.cache.PartialModelCache()
        # bumped by the optimality loop (synthesizer.py) each time the threshold tightens, so verdicts cached under a looser threshold are not reused
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

        self.first_full_assignment_checked = False

        # Reward properties only (see _model_check): PAYNT's default environment caps value iteration (Property.max_minmax_iterations), and on some models one
        # direction of a reward property converges so slowly that the capped result is off by orders of magnitude, which would make a refutation unsound. So
        # each direction is checked once, lazily, against policy iteration, and the method that works is used from then on. This is local to SMPMC: AR, CEGIS
        # and Hybrid only use such values for pruning and re-verify what they report as a concrete DTMC.
        self._reward_pi_environment: Any = None
        self._reward_convergence_verified: dict[bool, bool] = {}
        self._reward_use_policy_iteration: dict[bool, bool] = {}
        self._reward_verification_skip_warned: dict[bool, bool] = {}

    def check(self, fixed: dict[int, int], polarity: bool) -> TheoryResult | None:
        """:param fixed: {parameter_index: option} for every parameter currently decided in the partial
            model Z3 is exploring.
        :param polarity: True to check the `viable(...)` literal, False for `not viable(...)`.
        :returns: None if inconclusive (this partial assignment could still go either way), else a
            TheoryResult describing the refutation to hand back to Z3 as a learned conflict.
        """
        # The BitVec encoding has spare headroom beyond a parameter's option count (paynt/parameter_space/bitvec.py), so Z3 can transiently fix a variable to a
        # value outside its options. The in_range() constraint rejects such a branch on its own, and building a sub-MDP for it fails in stormpy's submodel
        # construction (deadlock state), so treat it as inconclusive.
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
            sub_mdp = self.colored_mdp.build_assignment(eta)
            paynt.utils.error_handling.require_markov_chain_for_robust_search(sub_mdp, self.policy_parameters)
        else:
            # No parent_selected_choices reuse hint: it assumes nested parameter spaces (true for AR's splitting), which Z3's CDCL backtracking does not give.
            sub_mdp, _selected_choices = self.colored_mdp.build(eta)
        self.mc_calls += 1
        if self.stat is not None:
            self.stat.iteration(sub_mdp)

        result = self._model_check(sub_mdp, alt=(polarity is False))

        # an infinite best case over several designs says nothing about them (see Property.best_case_is_unknown)
        if result.sat is polarity or (polarity and eta.size > 1 and self.prop.best_case_is_unknown(result.value)):
            self.cache.insert_inconclusive(fixed, polarity, self.epoch)
            return None

        # refuted: minimize the conflict by dropping every fixed parameter that isn't reachable (hence
        # irrelevant) in the induced sub-MDP
        relevant_parameters = self.relevant_parameters(eta, sub_mdp)
        conflict_parameters = [parameter for parameter in fixed if parameter in relevant_parameters]

        if self.stat is not None:
            # 'explored' estimate of this refutation: it rules out every complete assignment that agrees with it on conflict_parameters
            # This means this is an estimate not a precise count
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
            return set(self.colored_mdp.coloring.relevantParameters(eta.native, list(states)))
        assert self.state_to_parameters is not None
        relevant = stormpy.datastructures.BitVector(self.colored_mdp.parameter_space.num_parameters, False)
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

        TODO check if this is needed once new Storm is released with a convergence check for capped value iteration.
        """
        if not self.prop.reward:
            return sub_mdp.model_check_property(self.prop, alt=alt)

        if self._reward_use_policy_iteration.get(alt, False):
            return self._model_check_with_environment(sub_mdp, alt, self._pi_environment())

        vi_result = sub_mdp.model_check_property(self.prop, alt=alt)
        if self._reward_convergence_verified.get(alt, False):
            return vi_result

        if sub_mdp.model.nr_states >= _PI_VERIFICATION_MAX_STATES:
            # See _PI_VERIFICATION_MAX_STATES: one policy-iteration call is too expensive on a sub-MDP this large, so trust value iteration for this query.
            # _reward_convergence_verified[alt] stays unset: a later query in this direction can land on a much smaller sub-MDP (a full assignment does, via
            # build_assignment()'s DTMC), where the check is cheap and worth doing.
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
        # equal values agree, inf included, where the relative error would be nan
        if vi_value == pi_value or abs(vi_value - pi_value) / max(abs(pi_value), 1e-12) <= _REWARD_CONVERGENCE_RELATIVE_TOLERANCE:
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
        value = result.get_value(sub_mdp.initial_state)
        return paynt.specification.property_result.PropertyResult(self.prop, result, value)

    def _pi_environment(self) -> Any:
        if self._reward_pi_environment is None:
            env = stormpy.Environment()
            env.solver().min_max().method = stormpy.environments.MinMaxMethod.POLICY_ITERATION
            if not self.prop.sound:
                env.solver().min_max().maximal_number_of_iterations = self.prop.max_minmax_iterations
            self._reward_pi_environment = env
        return self._reward_pi_environment
