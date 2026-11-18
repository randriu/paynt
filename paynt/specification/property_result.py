from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from paynt.specification.property import Property, OptimalityProperty, Specification


class PropertyResult:
    def __init__(self, prop: Property, result: Any, value: Any, sat: bool = False):
        """:param value: None if there is no value: Storm decided the threshold without computing it, or the value of a conditional property is undefined
            (see paynt.utils.conditional)
        :param sat: without a value, whether the threshold is satisfied (not if the value is undefined)
        """
        self.result = result
        # only for a Cassandra-derived discounted-reward property, see Property.discount_correction_factor (a float factor, which an exact value would not take)
        if value is not None and prop.discount_factor_correction is not None:
            value = value * prop.discount_correction_factor
        self.value = value
        self.sat = prop.satisfies_threshold(self.value) if value is not None else sat
        self.improves_optimum: bool | None = None if not isinstance(prop, OptimalityProperty) else prop.improves_optimum(self.value)

    def __str__(self) -> str:
        if self.value is not None:
            return str(self.value)
        if self.result is None:
            return "undefined"
        return "satisfied" if self.sat else "violated"


class ConstraintsResult:
    """A list of constraint results.

    Note: some results might be None (not evaluated).
    """

    def __init__(self, results: list[Any]):
        self.results = results
        self.undecided_constraints = [i for i, result in enumerate(results) if result is not None and result.sat is None]

        result_sats = [result.sat for result in results if result is not None]
        if False in result_sats:
            self.sat: bool | None = False
        elif None in result_sats:
            self.sat = None
        else:
            self.sat = True

    def __str__(self) -> str:
        return ",".join([str(result) for result in self.results])


@dataclass
class SpecificationResult:
    constraints_result: ConstraintsResult | None = None
    optimality_result: Any = None

    def __str__(self) -> str:
        return str(self.constraints_result) + " : " + str(self.optimality_result)

    def accepting_dtmc(self, specification: Specification) -> tuple[bool, Any | None]:
        """Check whether the given specification is satisfied by this result's dtmc.

        :return: (1) whether specification is satisfied (dtmc is accepting)
        :return: (2) new optimal value associated with the accepting dtmc (can be None if no optimality)
        """
        assert self.constraints_result is not None
        if not self.constraints_result.sat:
            # constraints not sat
            return False, None

        if self.optimality_result is None:
            # constraints sat and no optimality
            return True, None

        if not self.optimality_result.improves_optimum:
            # constraints sat and optimality not sat
            return False, None

        # constraints sat and optimality sat
        return True, self.optimality_result.value

    def undecided_result(self) -> Any:
        if self.optimality_result is not None and self.optimality_result.can_improve:
            return self.optimality_result
        assert self.constraints_result is not None
        return self.constraints_result.results[self.constraints_result.undecided_constraints[0]]


@dataclass
class MdpPropertyResult:
    prop: Property
    primary: Any = None
    secondary: Any = None
    sat: bool | None = None
    primary_selection: list[list[int]] | None = None

    @property
    def minimizing(self) -> bool:
        return self.prop.minimizing

    def __str__(self) -> str:
        prim = str(self.primary)
        seco = str(self.secondary)
        if self.minimizing:
            return f"{prim} - {seco}"
        return f"{seco} - {prim}"


@dataclass
class MdpOptimalityResult(MdpPropertyResult):
    improving_assignment: Any = None
    improving_value: Any = None
    can_improve: bool | None = None


@dataclass
class MdpSpecificationResult(SpecificationResult):
    improving_assignment: Any = None
    improving_value: Any = None
    can_improve: bool | None = None

    def undecided_results(self) -> list[Any]:
        """Every still-relevant property result: undecided constraints (ascending index), then optimality if it can still improve.

        Deliberately constraints-first/optimality-last -- the opposite priority from undecided_result()'s single optimality-first pick, which is a different use
        case (Hybrid's priority- node heuristic) and is kept as-is. Used by the multi-constraint AR splitting logic (paper
        https://www.jair.org/index.php/jair/article/view/16593
        Section 3.3) to look at every property that
        could still inform which parameter to split on, not just one.
        """
        cr = self.constraints_result
        assert cr is not None
        results = [cr.results[i] for i in cr.undecided_constraints]
        if self.optimality_result is not None and self.optimality_result.can_improve:
            results.append(self.optimality_result)
        return results

    def evaluate(self, admissible_assignment: Any = None, admissible_value: Any = None) -> None:
        """Decide what the node holds.

        :param admissible_assignment: a design of the node that satisfies the whole specification (with an objective, improves the optimum), if one was found;
            when every design satisfies the constraints, it is the one to return, if one of them is not skipped (see
            paynt.utils.error_handling.DeadlockedDesigns)
        :param admissible_value: its value of the objective
        """
        self.improving_assignment = None
        self.improving_value = None
        self.can_improve = None

        cr = self.constraints_result
        opt = self.optimality_result
        assert cr is not None

        if cr.sat is False:
            self.can_improve = False
            return

        if opt is None:
            if admissible_assignment is not None:
                # one admissible design is all a specification without an objective asks for
                self.improving_assignment = admissible_assignment
                self.can_improve = False
            else:
                # undecided; or every design satisfies every constraint, but the one tried is skipped
                self.can_improve = True
            return

        if opt.improving_assignment is not None:
            # the design of the best case satisfies the constraints: no design of the node does better
            self.improving_assignment = opt.improving_assignment
            self.improving_value = opt.improving_value
        elif admissible_assignment is not None:
            # an admissible design improving the optimum, though better ones may remain
            self.improving_assignment = admissible_assignment
            self.improving_value = admissible_value
        self.can_improve = opt.can_improve
