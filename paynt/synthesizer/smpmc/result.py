from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import paynt.result

if TYPE_CHECKING:
    import paynt.parameter_space.parameter_space


@dataclass
class SmpmcResult(paynt.result.Result):
    # robust synthesis (--constraint exists_forall): the existentially quantified parameters fixed, the universally
    # quantified ones left at their full domain -- every member satisfies the specification; None otherwise
    # (Result.assignment is then one arbitrary member of this space)
    robust_assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
    # outcome of the independent AR check of robust_assignment (--smpmc-verify-robust); None if not checked
    robust_verified: bool | None = None
