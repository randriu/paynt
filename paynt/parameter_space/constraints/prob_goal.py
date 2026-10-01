"""ProbGoalConstraint: constrain the probability of reaching the specification's target to be either
strictly positive ("prob0" -- some way of resolving any leftover nondeterminism reaches it) or almost-sure
("prob1" -- every way of resolving leftover nondeterminism reaches it), encoded directly as Z3 assertions
over per-state reachability/min-step variables rather than via a Storm model-check call. This is a
*structural* property of the colored MDP itself, independent of any specific scheduler: even a fully
fixed parameter assignment can leave a state with more than one available action (an unlabelled/uncolored
choice is always available regardless of parameter values, alongside any colored ones -- see
payntbind/src/synthesis/coloring/Coloring.h's selectCompatibleChoices), so "every action makes progress"
(prob1) and "some action makes progress" (prob0) remain genuinely different constraints even after every
parameter is pinned down. That is exactly what the z3.And/z3.Or choice below distinguishes.

Ported from molehill's constraints/prob_goal.py (https://github.com/linusheck/molehill, GPL-3.0). The
prob=0 and prob=1 cases were two near-duplicate ~50-line code paths differing only in that one z3.And/
z3.Or swap on the final assertion per state; collapsed here into one parameterized implementation.
"""

from __future__ import annotations

from typing import Any

import z3

import paynt.model.model
import paynt.utils.error_handling
from paynt.parameter_space.constraints.constraint import Constraint, ConstraintContext

import logging

logger = logging.getLogger(__name__)


class ProbGoalConstraint(Constraint):
    name = "prob_goal"

    def __init__(self, prob: int):
        assert prob in (0, 1), "ProbGoalConstraint requires prob to be 0 (Prob>0) or 1 (Prob=1)"
        self.prob = prob

    def build(self, ctx: ConstraintContext) -> list[Any]:
        colored_mdp = ctx.colored_mdp
        paynt.utils.error_handling.require_pair_list_coloring(colored_mdp.coloring, "the prob0/prob1 constraints", "use another --constraint")
        model = colored_mdp.underlying_mdp
        transition_matrix = model.transition_matrix
        choice_to_assignment = colored_mdp.coloring.getChoiceToAssignment()

        assert len(model.initial_states) == 1, "ProbGoalConstraint only supports a single initial state"
        initial_state = model.initial_states[0]

        # this constraint is tied to one specific reachability target, so (like SynthesizerSMPMC itself)
        # it only makes sense against a single-property specification
        prop = ctx.task.get_property()
        target_states = paynt.model.model.ModelIndex.identify_target_states(model, prop)

        num_states = transition_matrix.nr_columns
        reach = [z3.Bool(f"__prob_goal_reach_{state}") for state in range(num_states)]
        min_step = [z3.Int(f"__prob_goal_min_step_{state}") for state in range(num_states)]

        # "every action must make progress" (prob=1, almost sure) vs "some action suffices" (prob=0,
        # possible) -- see this module's docstring for why these differ even once every parameter is fixed
        combine = z3.And if self.prob == 1 else z3.Or

        assertions: list[Any] = [step_var >= 0 for step_var in min_step]
        for state in range(num_states):
            if target_states.get(state):
                assertions.append(reach[state])
                assertions.append(min_step[state] == 0)
                continue

            per_choice_progress = []
            for choice in transition_matrix.get_rows_for_group(state):
                choice_assignment = z3.And([ctx.eq(parameter, option) for parameter, option in choice_to_assignment[choice]])

                successor_reach = []
                successor_min_step = []
                for entry in transition_matrix.get_row(choice):
                    if entry.value() == 0 or entry.column == state:
                        continue
                    successor_reach.append(reach[entry.column])
                    successor_min_step.append(min_step[entry.column])

                per_choice_progress.append(z3.Implies(choice_assignment, z3.Or(successor_reach)))
                # if this choice is selected and state is claimed reachable, min_step[state] must equal
                # exactly one more than the smallest reachable successor's min_step
                assertions.append(
                    z3.Implies(
                        z3.And(reach[state], choice_assignment),
                        z3.And(
                            z3.Or([min_step[state] == successor + 1 for successor in successor_min_step]),
                            z3.And([min_step[state] <= successor + 1 for successor in successor_min_step]),
                        ),
                    )
                )
            assertions.append(z3.Implies(reach[state], combine(per_choice_progress)))

        assertions.append(reach[initial_state])
        return ctx.base_clauses() + assertions
