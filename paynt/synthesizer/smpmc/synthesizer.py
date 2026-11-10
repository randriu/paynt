"""SynthesizerSMPMC: SMT-based synthesis over colored MDPs (arXiv:2511.08078)."""

from __future__ import annotations

import math
import re
from typing import Any

import z3

import paynt.colored_mdp
import paynt.parameter_space.constraints
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.synthesizer.search_node
import paynt.synthesizer.smpmc._augment
import paynt.synthesizer.smpmc._utils
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.smpmc.encoding
import paynt.synthesizer.smpmc.result
import paynt.synthesizer.smpmc.theory
import paynt.synthesizer.synthesizer
import paynt.task
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)

#: Z3's timeout parameter is an unsigned 32-bit number of milliseconds
_MAX_Z3_TIMEOUT_MS = 2**32 - 1
#: what solver.reason_unknown() says when check() was stopped by the timeout set by SynthesizerSMPMC._limit_solver_time, or interrupted by the theory
#: (ColoredMdpTheory.time_is_up)
_TIMEOUT_REASONS = ("timeout", "canceled", "interrupted")


class SynthesizerSMPMC(paynt.synthesizer.synthesizer.Synthesizer):
    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask):
        super().__init__(colored_mdp, task)
        spec = task.specification
        paynt.utils.error_handling.require_single_property(spec)
        self.prop: paynt.specification.property.Property = spec.optimality if spec.optimality is not None else spec.constraints[0]
        # SMPMC always needs a constraint, as it supplies the viable(...) clause that the theory integration depends on, so it defaults to "exists"
        self.constraint = paynt.parameter_space.constraints.build_constraint(task.constraint_name or "exists")

        # robust synthesis: the universally quantified parameters, and the robust space found by the last run
        self.forall_parameters: list[int] = []
        self.robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
        if self.constraint.name == "exists_forall":
            self._init_forall_parameters()

    def _init_forall_parameters(self) -> None:
        if self.colored_mdp.feature_kind == "family":
            self.colored_mdp, self.forall_parameters = paynt.synthesizer.smpmc._augment.add_policy_parameters(self.colored_mdp)
        parameter_space = self.colored_mdp.parameter_space
        if self.task.forall_pattern is not None:
            pattern = re.compile(self.task.forall_pattern)
            self.forall_parameters = [
                parameter for parameter in range(parameter_space.num_parameters) if pattern.search(parameter_space.parameter_name(parameter))
            ]
            if not self.forall_parameters:
                raise ValueError(f"--smpmc-forall {self.task.forall_pattern!r} matches no parameter")
        elif not self.forall_parameters:
            raise ValueError("--constraint exists_forall needs --smpmc-forall to select the universally quantified parameters")
        forall_names = [parameter_space.parameter_name(parameter) for parameter in self.forall_parameters]
        num_existential = parameter_space.num_parameters - len(self.forall_parameters)
        logger.info(f"robust synthesis: {num_existential} existential parameters, universally quantified: {forall_names}")

    @property
    def method_name(self) -> str:
        return "SMPMC"

    def synthesize_one(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        encoding = paynt.synthesizer.smpmc.encoding.ParameterBitVecEncoding(node.parameter_space)
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(self.colored_mdp, self.prop, self.stat, self.forall_parameters)
        theory.time_is_up = self._time_is_up

        sorts = [variable.sort() for variable in encoding.variables]
        viable = z3.PropagateFunction("viable", *sorts, z3.BoolSort())
        ctx = paynt.parameter_space.constraints.ConstraintContext(
            colored_mdp=self.colored_mdp,
            parameter_space=node.parameter_space,
            task=self.task,
            variables=encoding.variables,
            width=encoding.width,
            viable=viable,
            forall_parameters=self.forall_parameters,
        )
        clauses = self.constraint.build(ctx)
        # the parameters a witness fixes: all of them, or the policy of a robust search
        policy_parameters = [parameter for parameter in range(node.parameter_space.num_parameters) if parameter not in self.forall_parameters]
        # clauses ruling out witnesses whose value is known, see below
        exclusions: list[Any] = []

        def new_solver() -> tuple[Any, Any]:
            solver = z3.Solver()
            # Register the propagator before adding the clauses, so that `created` fires for the viable(...) terms as they are asserted. Keep it alive as long
            # as the solver is used: it owns the native callback registration, so garbage-collecting it mid-search is a use-after-free.
            propagator = paynt.synthesizer.smpmc.theory.SmpmcPropagator(solver, None, theory, encoding.name_to_parameter)
            solver.add(*clauses, *exclusions)
            return solver, propagator

        solver, _propagator = new_solver()
        optimality = self.task.specification.optimality
        while not self.resource_limit_reached():
            self._limit_solver_time(solver)
            result = solver.check()
            if theory.failure is not None:
                raise theory.failure
            if result == z3.unsat:
                break
            if self._time_is_up():
                logger.info("time limit reached, aborting...")
                break
            if result == z3.unknown:
                if solver.reason_unknown() in _TIMEOUT_REASONS:
                    logger.info("time limit reached, aborting...")
                else:
                    logger.warning(f"SMPMC returned unknown: {solver.reason_unknown()}")
                break

            assignment = encoding.extract_assignment(solver.model(), node.parameter_space, self.forall_parameters)

            if optimality is None:
                if self.forall_parameters:
                    self.robust_assignment = assignment
                self.best_assignment = assignment
                break

            if self.forall_parameters:
                value, exact = self._worst_case_of(theory, assignment, policy_parameters, optimality)
            else:
                value, exact = self._value_of(assignment), True
            # Record the witness as the new optimum if it beats the threshold, and exclude it from Z3's search if its value is final (`exact`).
            # A robust witness's value may be just a bound on its worst case, so it can still turn out better: do not exclude it then.
            improves = optimality.improves_optimum(value)
            # new optimum: record it and tighten the threshold
            if improves:
                self.best_assignment = assignment
                self.best_assignment_value = value
                if self.forall_parameters:
                    self.robust_assignment = assignment
                optimality.update_optimum(value)
                # the tighter threshold invalidates some cached verdicts (see PartialModelCache)
                theory.epoch += 1
            if exact:
                # the value is final: exclude the witness (in a robust search, its whole policy) from Z3's search, kept in `exclusions` for rebuilt solvers
                exclusions.append(encoding.exclude(assignment, policy_parameters))
            if improves and self.forall_parameters:
                # robust: Z3 may have learned clauses that the tighter threshold falsifies and cannot retract, so start a new solver (the theory cache is kept)
                solver, _propagator = new_solver()
            elif exact:
                # otherwise just add the new exclusion to the current solver
                solver.add(exclusions[-1])

        self.explored = node.parameter_space.size
        return self.best_assignment

    def _time_is_up(self) -> bool:
        remaining = self.time_remaining()
        return remaining is not None and remaining <= 0

    def _limit_solver_time(self, solver: Any) -> None:
        """Make the next solver.check() give up, answering unknown, once the time limit is reached."""
        remaining = self.time_remaining()
        if remaining is not None:
            solver.set("timeout", min(max(math.ceil(remaining * 1000), 1), _MAX_Z3_TIMEOUT_MS))

    def _worst_case_of(
        self,
        theory: paynt.synthesizer.smpmc.checker.ColoredMdpTheory,
        robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace,
        policy_parameters: list[int],
        optimality: paynt.specification.property.OptimalityProperty,
    ) -> tuple[Any, bool]:
        """The worst case over the environments of the policy robust_assignment fixes, and whether it is exact rather than a bound."""
        policy = {parameter: robust_assignment.parameter_options(parameter)[0] for parameter in policy_parameters}
        bound = theory.worst_case_bound(policy)
        if bound is not None and optimality.improves_optimum(bound.value):
            return bound.value, bound.exact
        logger.warning("no usable worst-case bound for the robust policy, computing its worst case by AR")
        return paynt.synthesizer.smpmc._utils.worst_case_by_ar(self.colored_mdp, optimality, robust_assignment), True

    def run(self, optimum_threshold: Any = None) -> paynt.synthesizer.smpmc.result.SmpmcResult:
        """Also reports the robust space: synthesize() collapses best_assignment to one member of it (and double-checks that one)."""
        self.robust_assignment = None
        result = super().run(optimum_threshold)
        robust_verified = None
        if self.robust_assignment is not None:
            if result.value is None:
                logger.info(f"robust assignment, every member satisfies the specification: {self.robust_assignment}")
            else:
                logger.info(f"robust assignment, worst case {result.value} over its members: {self.robust_assignment}")
            if self.task.verify_robust:
                robust_verified = paynt.synthesizer.smpmc._utils.verify_robust(self.colored_mdp, self.task, self.robust_assignment, result.value)
                logger.info(f"robustness verified by AR: {robust_verified}")
        return paynt.synthesizer.smpmc.result.SmpmcResult(
            success=result.success,
            value=result.value,
            assignment=result.assignment,
            selected_choices=result.selected_choices,
            robust_assignment=self.robust_assignment,
            robust_verified=robust_verified,
        )

    def _value_of(self, assignment: paynt.parameter_space.parameter_space.ParameterSpace) -> Any:
        model = self.colored_mdp.build_assignment(assignment)
        return model.model_check_property(self.prop).value
