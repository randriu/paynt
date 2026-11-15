"""The errors for what a coloring, a specification, a sketch or a synthesis method cannot do, kept out of the classes that come across them so that those stay
short: designs that leave a state without a choice, what an incomplete coloring does not support (a full parameter assignment that leaves several choices
enabled, see paynt.colored_mdp), a general coloring where the (parameter, option) pairs are needed, a hole where a sketch cannot have one (the reward of a
state, a label), a specification that an engine does not support, and a method that a feature does not run."""

from __future__ import annotations

import collections
import contextlib
import json
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any

import stormpy

import paynt.utils.coloring

if TYPE_CHECKING:
    import paynt.colored_mdp
    import paynt.model.model
    import paynt.parameter_space.parameter_space
    import paynt.specification.property


class StateWithoutChoice(RuntimeError):
    """The choices from which a sub-MDP is built leave a state that they reach without a choice (see paynt.model.model.SubmodelBuilder.restrict)."""


class DeadlockedDesigns(ValueError):
    """Every design of a parameter space leaves a state that it reaches without a choice (a deadlock).

    Such a design is skipped: an engine leaves it out, with a
    warning, as if it were not in the parameter space.
    """

    def __init__(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, state: int, valuation: str, states: list[int]):
        """:param state: a state of the underlying MDP that the designs reach and leave without a choice
        :param valuation: its variables, as text
        :param states: the states whose choices make it so: every design that selects the same choices there deadlocks too (see
            ColoredMdp.relevant_parameters)
        """
        self.parameter_space = parameter_space
        self.state = state
        self.states = states
        designs = f"design {parameter_space}" if parameter_space.size == 1 else f"the {parameter_space.size} designs of {parameter_space}"
        where = f"state {state} ({valuation})" if valuation else f"state {state}"
        super().__init__(f"skipping {designs}: {where} is reachable and keeps no choice")


@contextlib.contextmanager
def raise_deadlocked_design(
    colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
) -> Generator[None]:
    """Build the model of a design in the block; if a state it reaches keeps no choice, raise DeadlockedDesigns.

    :param choices: the choices of the underlying MDP that the coloring enabled for parameter_space (one design), from which the block builds
    """
    try:
        yield
    except StateWithoutChoice:
        raise _deadlocked_design(colored_mdp, choices, parameter_space) from None


def choices_avoiding_deadlocks(
    colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
) -> tuple[Any, list[int]]:
    """Of the choices of parameter_space, which leave a state they reach without a choice, those that keep away from such states.

    A design that keeps a choice in every state it reaches takes only such choices, so the sub-MDP of these holds every design of parameter_space that is not
    skipped.

    :return: (1) those choices
    :return: (2) the states that the choices of parameter_space reach, whose choices decide (1)
    :raises DeadlockedDesigns: if parameter_space is a single design, or if no design keeps away from such states
    """
    deadlocked = _deadlocked_design(colored_mdp, choices, parameter_space)
    if parameter_space.size == 1:
        raise deadlocked
    avoiding, reached = _choices_avoiding_states_without_choice(colored_mdp, choices)
    if avoiding is None:
        # every policy reaches such a state, so every design does: what decides it are the choices of the states reached
        raise DeadlockedDesigns(parameter_space, deadlocked.state, _valuation(colored_mdp, deadlocked.state), reached)
    return avoiding, reached


def _deadlocked_design(
    colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
) -> DeadlockedDesigns:
    """DeadlockedDesigns for the choices of parameter_space, which leave a state they reach without a choice."""
    path = _path_to_state_without_choice(colored_mdp, choices)
    assert path is not None, "the choices leave no state they reach without a choice"
    # a design that selects the same choices in the states on the path reaches the last one as well, and keeps no choice there either
    return DeadlockedDesigns(parameter_space, path[-1], _valuation(colored_mdp, path[-1]), path)


def _path_to_state_without_choice(colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any) -> list[int] | None:
    """A shortest path from the initial state, through the choices, to a state that keeps none of them: its states, that one last.

    None if every state that the choices reach keeps one.
    """
    row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
    predecessor: dict[int, int | None] = {int(state): None for state in colored_mdp.underlying_mdp.initial_states}
    queue = collections.deque(predecessor)
    while queue:
        state = queue.popleft()
        enabled = [choice for choice in range(row_groups[state], row_groups[state + 1]) if choices[choice]]
        if not enabled:
            path = [state]
            while (previous := predecessor[path[-1]]) is not None:
                path.append(previous)
            return path[::-1]
        for choice in enabled:
            for destination in colored_mdp.choice_destinations[choice]:
                if destination not in predecessor:
                    predecessor[destination] = state
                    queue.append(destination)
    return None


def _choices_avoiding_states_without_choice(colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any) -> tuple[Any, list[int]]:
    """The largest subset of the choices that never leads to a state without one of them: a policy that takes only these never reaches such a state, and one
    that takes another one can. Only the states that the choices reach from the initial state count.

    :return: (1) the subset, as a choice mask, or None if the initial state keeps none of it
    :return: (2) the states that the choices reach
    """
    row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
    enabled: dict[int, list[int]] = {}
    queue = collections.deque(int(state) for state in colored_mdp.underlying_mdp.initial_states)
    reached = set(queue)
    while queue:
        state = queue.popleft()
        enabled[state] = [choice for choice in range(row_groups[state], row_groups[state + 1]) if choices[choice]]
        for choice in enabled[state]:
            for destination in colored_mdp.choice_destinations[choice]:
                if destination not in reached:
                    reached.add(destination)
                    queue.append(destination)

    # remove the choices into a state without one until none is left
    source = {choice: state for state, state_choices in enabled.items() for choice in state_choices}
    into: dict[int, list[int]] = collections.defaultdict(list)
    for choice in source:
        for destination in colored_mdp.choice_destinations[choice]:
            into[destination].append(choice)
    kept = {state: len(state_choices) for state, state_choices in enabled.items()}
    removed: set[int] = set()
    stuck = [state for state, count in kept.items() if count == 0]
    while stuck:
        for choice in into[stuck.pop()]:
            if choice in removed:
                continue
            removed.add(choice)
            kept[source[choice]] -= 1
            if kept[source[choice]] == 0:
                stuck.append(source[choice])

    if any(kept[int(state)] == 0 for state in colored_mdp.underlying_mdp.initial_states):
        return None, sorted(reached)
    avoiding = stormpy.datastructures.BitVector(colored_mdp.underlying_mdp.nr_choices, False)
    for choice in source:
        if choice not in removed:
            avoiding.set(choice, True)
    return avoiding, sorted(reached)


def _valuation(colored_mdp: paynt.colored_mdp.ColoredMdp, state: int) -> str:
    """The variables of a state of the underlying MDP, as text; empty without state valuations."""
    model = colored_mdp.underlying_mdp
    if not model.has_state_valuations():
        return ""
    valuation = json.loads(str(model.state_valuations.to_json(state)))
    return ", ".join(f"{variable}={value}" for variable, value in valuation.items() if "_loc_prism2jani_" not in variable)


def require_eventually(prop: paynt.specification.property.Property) -> None:
    """Raise NotImplementedError if prop is an until formula that is checked as it is, for code that needs a target to reach.

    An until formula `phi1 U phi2` is checked as `F phi2` when no state where neither side holds can reach phi2 (see Property.transform_until_to_eventually);
    otherwise only model checking supports it.
    """
    if prop.is_until:
        raise NotImplementedError(
            f"{prop} is checked as an until formula, which this needs as reaching a target: a state where neither side holds can reach the right side "
            "(an until formula is read as reaching its right side when Storm makes such states absorbing, as it does for a model built for this formula alone "
            "with atomic sides)"
        )


def require_single_property_on_nondeterministic_model(model: paynt.model.model.Mdp, specification: paynt.specification.property.Specification) -> None:
    """Raise NotImplementedError if the model still has nondeterminism and the specification has several properties.

    A full parameter assignment of an incomplete coloring, or of an MDP family (whose policy is not a parameter), leaves several choices enabled, and each
    property is checked in its own direction, with the best resolution of those choices for it: several properties would each be given a policy of their own
    instead of a common one.
    """
    if not model.is_deterministic and specification.num_properties > 1:
        raise NotImplementedError(
            f"cannot check {specification.num_properties} properties on a model that still has nondeterminism: when a parameter assignment leaves several "
            "choices enabled (an incomplete coloring, or an MDP family, whose policy is not a parameter), each property would be given a policy of its own for "
            "them, so only a single property is supported"
        )


def require_no_maximizing_reward(specification: paynt.specification.property.Specification) -> None:
    """Raise NotImplementedError if the specification maximizes a reward (an objective, or a constraint R>= or R>), which CEGIS, and hybrid with it, does not
    support."""
    if specification.contains_maximizing_reward_properties:
        raise NotImplementedError("CEGIS (also inside hybrid) does not support maximizing reward properties; use --method ar, onebyone or smpmc")


def require_single_property(specification: paynt.specification.property.Specification) -> None:
    """Raise NotImplementedError if the specification has several properties, which SMPMC does not support."""
    if specification.num_properties > 1:
        raise NotImplementedError(
            "SMPMC currently supports a single property: either one plain threshold constraint or one optimality objective, not both together and not "
            "several constraints at once"
        )


def require_markov_chain_for_robust_search(model: paynt.model.model.Mdp, policy_parameters: list[int] | None) -> None:
    """Raise NotImplementedError if a robust search (policy_parameters is not None) is given the model of a full assignment that still has nondeterminism.

    With the policy chosen before the environment, `viable` is judged by the best case and `not viable` by the worst, and in the gap between them neither
    literal is ever refuted: Z3 would end with "unknown". The choices left would have to be policy parameters.
    """
    if policy_parameters is not None and not model.is_deterministic:
        raise NotImplementedError(
            "robust synthesis does not support an incomplete coloring: a full parameter assignment leaves several choices enabled, and the policy "
            "has to be chosen before the environment, so it must be given by parameters"
        )


def require_pair_list_coloring(coloring: Any, what: str, alternative: str | None = None) -> None:
    """Raise NotImplementedError if coloring is a general coloring.

    For code that reads the coloring's (parameter, option) pairs back (getChoiceToAssignment, collectHoleOptions, ...): a general coloring is an arbitrary
    formula per choice, so it has none.

    :param what: what needs the pairs, as a noun phrase, e.g. "the policy tree"
    :param alternative: what to do instead, if anything
    """
    if paynt.utils.coloring.is_general_coloring(coloring):
        message = f"cannot use a general coloring for {what}: it has no (parameter, option) pairs to read back"
        raise NotImplementedError(message if alternative is None else f"{message}; {alternative}")


def require_supported_method(method: str, supported: Sequence[str], what: str) -> None:
    """Raise ValueError unless method is one of those that `what` (a noun phrase, e.g. "decision-tree synthesis") can run."""
    if method not in supported:
        raise ValueError(f"{what} supports method {' or '.join(repr(name) for name in supported)}, got {method!r}")


def require_method_ar(method: str, what: str) -> None:
    """Raise ValueError unless method is "ar", for what (a noun phrase, e.g. "--dtnest") that always searches with AR over ColoringSmt."""
    if method != "ar":
        raise ValueError(f"{what} does not support method {method!r}: it always uses AR over ColoringSmt, use method 'ar'")


def _identifiers(expression: Any) -> set[str]:
    """The identifiers in an expression of the JSON form of a JANI model: the strings of the expression other than the operators and the comments, which include
    the names of the functions that it calls."""
    found: set[str] = set()
    stack = [expression]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, dict):
            stack.extend(value for key, value in item.items() if key not in ("op", "comment"))
        elif isinstance(item, list):
            stack.extend(item)
    return found


def _holes_of(model: dict[str, Any], holes: Sequence[str]) -> dict[str, set[str]]:
    """For every hole, constant and function of the JANI model (in JSON form) that depends on a hole, the holes that it depends on: a hole on itself, a constant
    or a function (a PRISM formula) on the holes of its definition."""
    depends_on: dict[str, set[str]] = {hole: {hole} for hole in holes}
    definitions = [(constant["name"], constant["value"]) for constant in model.get("constants", []) if "value" in constant]
    definitions += [(function["name"], function["body"]) for function in model.get("functions", [])]
    changed = True
    while changed:
        changed = False
        for name, definition in definitions:
            reached = set().union(*(depends_on.get(identifier, set()) for identifier in _identifiers(definition)))
            known = depends_on.get(name, set())
            if not reached <= known:
                depends_on[name] = known | reached
                changed = True
    return depends_on


def require_no_holes_in_state_values(jani: Any, holes: Sequence[str]) -> None:
    """Raise ValueError if the reward of a state, or a label, depends on a hole of the sketch.

    The unfolder substitutes the options of the holes into the edges of the JANI model, but the reward of a state and a label are transient values of its
    location, which it does not touch: the hole would stay in them, undefined, and the reward or the label would come out wrong without any error.

    :param jani: the JANI model of the sketch, with the holes as undefined constants
    :param holes: the names of the holes
    """
    model = json.loads(str(jani))
    depends_on = _holes_of(model, holes)
    kinds = {
        variable["name"]: "label" if variable.get("type") == "bool" else "state reward" for variable in model.get("variables", []) if variable.get("transient")
    }
    for automaton in model.get("automata", []):
        for location in automaton.get("locations", []):
            for transient_value in location.get("transient-values", []):
                used = sorted(set().union(*(depends_on.get(identifier, set()) for identifier in _identifiers(transient_value["value"]))))
                if used:
                    name = transient_value["ref"]
                    kind = kinds.get(name, "state reward")
                    instead = (
                        "Use the hole in the reward of a transition ([action] guard : value;) instead."
                        if kind == "state reward"
                        else "Put the condition into the property, or into the guards of the commands, instead."
                    )
                    raise ValueError(
                        f"the {kind} {name!r} depends on the hole{'s' if len(used) > 1 else ''} {', '.join(used)}, which is not supported: PAYNT substitutes "
                        f"the options of a hole into the transitions of the model, which the reward of a state and a label are not on, so the hole would stay "
                        f"in it unresolved. {instead}"
                    )
