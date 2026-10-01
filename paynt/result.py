"""The final result of a completed synthesis run for a SynthesisTask.

Feature-specific results subclass this to add their own fields -- e.g. paynt.dt.result.DtResult adds the synthesized tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import paynt.parameter_space.parameter_space


@dataclass
class Result:
    success: bool
    # the achieved optimum, or None if the specification has no optimality objective
    value: float | None = None
    # the winning parameter assignment, or None if none was found
    # (some feature results -- e.g. paynt.mdp_family.policy_tree_synthesizer's policy-tree result -- have no
    # single assignment/value at all, since they represent a set of region-specific policies rather than
    # one answer; those leave both fields None, which the caller should read as "not applicable", not
    # "not found")
    assignment: paynt.parameter_space.parameter_space.ParameterSpace | None = None
    # the choices of the underlying MDP the solution stands for, as a mask (a stormpy BitVector), or None if there is no assignment: the ones the assignment
    # enables and, in a state it leaves several choices (an incomplete coloring, see paynt.colored_mdp), the one the best policy takes -- so the uncolored
    # choices are in it too. Exactly one choice for every state the solution reaches, i.e. its Markov chain:
    # ColoredMdp.build_from_choice_mask(selected_choices).
    selected_choices: Any = None
