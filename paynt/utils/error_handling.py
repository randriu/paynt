"""The errors for what a coloring, a specification, a sketch or a synthesis method cannot do, kept out of the classes that come across them so that those stay
short: a state that a coloring leaves without a choice, what an incomplete coloring does not support (a full parameter assignment that leaves several choices
enabled, see paynt.colored_mdp), a general coloring where the (parameter, option) pairs are needed, a hole where a sketch cannot have one (the reward of a
state, a label), and a method that a feature does not run."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any

import paynt.utils.coloring

if TYPE_CHECKING:
    import paynt.colored_mdp
    import paynt.model.model
    import paynt.parameter_space.parameter_space
    import paynt.specification.property


@contextlib.contextmanager
def explain_state_without_choice(
    colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
) -> Generator[None]:
    """Build a sub-MDP of the colored MDP in the block; if that fails because a state is left without a choice, raise an error that says which state and what
    the coloring did, instead of Storm's.

    :param choices: the choices of the underlying MDP that the coloring enabled for parameter_space, from which the block builds
    """
    try:
        yield
    except RuntimeError as error:
        raise _state_without_choice(error, colored_mdp, choices, parameter_space) from None


def _state_without_choice(
    error: RuntimeError, colored_mdp: paynt.colored_mdp.ColoredMdp, choices: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
) -> Exception:
    """The error to raise for a failure to build the sub-MDP of choices: if Storm found a state without a choice, and it is a reachable one, an error that says
    which state; otherwise the original."""
    if "deadlock" not in str(error):
        return error
    row_groups = colored_mdp.underlying_mdp.nondeterministic_choice_indices
    reached = {int(state) for state in colored_mdp.underlying_mdp.initial_states}
    frontier = list(reached)
    while frontier:
        state = frontier.pop()
        enabled = [choice for choice in range(row_groups[state], row_groups[state + 1]) if choices[choice]]
        if not enabled:
            return ValueError(
                f"the coloring enables no choice in state {state}, which is reachable, for the parameter space {parameter_space}: every reachable "
                "state must keep at least one choice. Restrict the options of the parameters that color its choices to those that leave one, and check a "
                "hand-written coloring with ColoringBuilder.check_definition(..., complete=False)"
            )
        for choice in enabled:
            for destination in colored_mdp.choice_destinations[choice]:
                if destination not in reached:
                    reached.add(destination)
                    frontier.append(destination)
    return error


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
