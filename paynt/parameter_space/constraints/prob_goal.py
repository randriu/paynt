"""ProbGoalConstraint: the specification's target must be reached from the initial state with positive probability under some way of resolving leftover
nondeterminism ("prob0"), or with probability 1 under every way ("prob1").

Encoded as Z3 assertions over per-state variables, without calling Storm.
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

        # this constraint is tied to one specific reachability target,
        # so it only makes sense against a single-property specification
        prop = ctx.task.get_property()
        target_states = paynt.model.model.ModelIndex.identify_target_states(model, prop)

        num_states = transition_matrix.nr_columns
        # reach[s]: the target is reached from s with positive probability under some (prob0) / every (prob1) resolution of the nondeterminism;
        # rank[s] decreases along the path that witnesses it, so that a cycle cannot witness itself
        # the lists given to And/Or below may be empty, which Z3 would build in its global context without z3_context
        z3_context = ctx.z3_context
        reach = [z3.Bool(f"__prob_goal_reach_{state}", ctx=z3_context) for state in range(num_states)]
        rank = [z3.Int(f"__prob_goal_rank_{state}", ctx=z3_context) for state in range(num_states)]
        # visited[s] (prob1): s can be visited before the target is reached
        visited = [z3.Bool(f"__prob_goal_visited_{state}", ctx=z3_context) for state in range(num_states)]

        assertions: list[Any] = [state_rank >= 0 for state_rank in rank]
        for state in range(num_states):
            if target_states.get(state):
                assertions.append(reach[state])
                continue

            per_choice_progress = []
            for choice in transition_matrix.get_rows_for_group(state):
                enabled = z3.And([ctx.eq(parameter, option) for parameter, option in choice_to_assignment[choice]], z3_context)
                successors = [entry.column for entry in transition_matrix.get_row(choice) if entry.value != 0 and entry.column != state]
                progress = z3.Or([z3.And(reach[successor], rank[successor] < rank[state]) for successor in successors], z3_context)
                if self.prob == 0:
                    per_choice_progress.append(z3.And(enabled, progress))
                else:
                    per_choice_progress.append(z3.Implies(enabled, progress))
                    assertions.append(z3.Implies(z3.And(visited[state], enabled), z3.And([visited[successor] for successor in successors], z3_context)))
            if self.prob == 0:
                assertions.append(z3.Implies(reach[state], z3.Or(per_choice_progress, z3_context)))
            else:
                assertions.append(z3.Implies(reach[state], z3.And(per_choice_progress, z3_context)))
                # probability 1: no state from which the target may be missed can be visited
                assertions.append(z3.Implies(visited[state], reach[state]))

        assertions.append(reach[initial_state] if self.prob == 0 else visited[initial_state])
        return ctx.base_clauses() + assertions
