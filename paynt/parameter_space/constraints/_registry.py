"""Name -> Constraint class lookup.

Each engine decides its own default when task.constraint_name is unset: SMPMC always needs *some* constraint (it's what supplies the viable(...) clause), so it
defaults to "exists"; CEGIS's pre-existing behavior is no constraint at all, so it only builds one when the user explicitly asks for it.
"""

from __future__ import annotations

# direct name imports throughout this file for the same reason as exists.py: everything here runs during
# paynt/parameter_space/constraints/__init__.py's own initialization, before paynt.parameter_space.constraints
# is bound as an attribute of paynt.parameter_space, so dotted attribute access on that path would fail
from collections.abc import Callable

from paynt.parameter_space.constraints.constraint import Constraint
from paynt.parameter_space.constraints.exists import ExistsConstraint
from paynt.parameter_space.constraints.exists_forall import ExistsForallConstraint
from paynt.parameter_space.constraints.costs import CostsConstraint
from paynt.parameter_space.constraints.prob_goal import ProbGoalConstraint

CONSTRAINTS: dict[str, Callable[[], Constraint]] = {
    "exists": ExistsConstraint,
    "exists_forall": ExistsForallConstraint,
    "costs": CostsConstraint,
    "prob0": lambda: ProbGoalConstraint(prob=0),
    "prob1": lambda: ProbGoalConstraint(prob=1),
}


def build_constraint(name: str) -> Constraint:
    if name not in CONSTRAINTS:
        raise ValueError(f"unknown constraint {name!r}; available: {sorted(CONSTRAINTS)}")
    return CONSTRAINTS[name]()
