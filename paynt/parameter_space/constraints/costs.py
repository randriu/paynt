"""CostsConstraint: bound the total cost of the chosen parameter options by a threshold, where per- (parameter, option) costs are read from a sketch.costs
sidecar file.

Ported from molehill's constraints/costs.py (https://github.com/linusheck/molehill, GPL-3.0). The sidecar
format is unchanged: one "parameter_name option_index cost" line per (parameter, option) pair, enumerated
in the same order as `for parameter in range(num_parameters): for option in range(num_options(parameter))`
-- molehill's own strict line-by-line assertions are kept, since a silently-misaligned costs file would
otherwise produce a constraint that looks fine but bounds the wrong thing.
"""

from __future__ import annotations

from typing import Any

import z3

from paynt.parameter_space.constraints.constraint import Constraint, ConstraintContext

import logging

logger = logging.getLogger(__name__)


class CostsConstraint(Constraint):
    name = "costs"

    def build(self, ctx: ConstraintContext) -> list[Any]:
        assert ctx.task.costs_file_path is not None, "the costs constraint requires a sketch.costs file alongside the sketch"
        assert ctx.task.costs_threshold is not None, "the costs constraint requires --costs-threshold"

        with open(ctx.task.costs_file_path) as f:
            lines = f.readlines()

        parameter_space = ctx.parameter_space
        cost_vars = []
        cost_assertions: list[Any] = []
        line_index = 0
        for parameter in range(parameter_space.num_parameters):
            parameter_name = parameter_space.parameter_name(parameter)
            for option in range(parameter_space.parameter_num_options_total(parameter)):
                line_parameter, line_option, cost_value = lines[line_index].split()
                line_index += 1
                assert line_parameter == parameter_name, f"sketch.costs line {line_index}: expected parameter {parameter_name!r}, got {line_parameter!r}"
                assert int(line_option) == option, f"sketch.costs line {line_index}: expected option {option}, got {line_option}"

                cost_var = z3.Int(f"__cost_{parameter_name}_{option}")
                cost_vars.append(cost_var)
                cost_assertions.append(z3.If(ctx.eq(parameter, option), cost_var == int(cost_value), cost_var == 0))

        cost_assertions.append(z3.Sum(cost_vars) <= ctx.task.costs_threshold)
        return ctx.base_clauses() + cost_assertions
