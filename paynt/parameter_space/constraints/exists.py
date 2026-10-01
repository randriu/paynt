"""ExistsConstraint: the plain existential case -- every parameter within its currently-allowed options, and (for SMPMC) a solution must be viable.

This was Phase 1's only constraint, inlined directly in paynt/synthesizer/smpmc/synthesizer.py; extracted here once CEGIS became a second consumer of the shared
constraint framework.
"""

from __future__ import annotations

from typing import Any

# direct name imports, not "import paynt.parameter_space.constraints.constraint" + dotted access: this
# module is imported from within paynt/parameter_space/constraints/__init__.py's own initialization (via
# _registry), so paynt.parameter_space.constraints is not yet bound as an attribute of paynt.parameter_space
# at that point -- only a direct sys.modules-based name import survives the partial initialization
from paynt.parameter_space.constraints.constraint import Constraint, ConstraintContext


class ExistsConstraint(Constraint):
    name = "exists"

    def build(self, ctx: ConstraintContext) -> list[Any]:
        return ctx.base_clauses()
