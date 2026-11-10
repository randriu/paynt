"""The reference answers for tiny sketches: every design built and model-checked on its own, in exact arithmetic, by Storm alone.

The holes of the sketch are replaced by constants, one design at a time, so none of PAYNT's parsing, coloring or search is involved. A design of an MDP sketch
is an MDP, and each property is checked in its own direction: the best policy for it. As in PAYNT, a design whose expected reward is infinite (it does not reach
the goal for sure) satisfies no reward property, and a design whose holes leave a state without a command that other designs have there is skipped.
"""

from __future__ import annotations

import functools
import itertools
import math
import operator
import os
import re
import tempfile
from dataclasses import dataclass
from fractions import Fraction

import stormpy

HOLE = re.compile(r"^\s*hole\s+(\w+)\s+(\w+)\s+in\s+\{(.*?)\}\s*;")
COMPARISONS = {
    stormpy.logic.ComparisonType.LESS: operator.lt,
    stormpy.logic.ComparisonType.LEQ: operator.le,
    stormpy.logic.ComparisonType.GREATER: operator.gt,
    stormpy.logic.ComparisonType.GEQ: operator.ge,
}


def hole_options(text):
    """Option list of `{0..3}`, `{0..6:2}` or `{1,2,4}`."""
    if ".." in text:
        bounds, _, step = text.partition(":")
        low, high = bounds.split("..")
        return [str(option) for option in range(int(low), int(high) + 1, int(step or 1))]
    return [option.strip() for option in text.split(",")]


def number(label):
    """An option label as a number, so that 0.5 and 1/2 are the same option."""
    try:
        return Fraction(label)
    except ValueError:
        return label


@dataclass(frozen=True)
class Property:
    text: str
    reward: bool
    maximizing: bool
    # for a constraint, its comparison and threshold; None for the objective
    compare: object = None
    threshold: Fraction | None = None

    @property
    def is_objective(self):
        return self.compare is None

    def valid(self, value):
        return not (self.reward and value == math.inf)

    def holds(self, value):
        return self.valid(value) and self.compare(value, self.threshold)


@dataclass(frozen=True)
class Design:
    options: tuple
    # one value per property: a Fraction, or math.inf for an infinite reward
    values: tuple
    # the valuations of the states without a command, and of the others
    deadlocks: frozenset = frozenset()
    live: frozenset = frozenset()

    def __str__(self):
        return ", ".join(f"{hole}={option}" for hole, option in self.options)


def _formula(prop):
    """The property without its bound, in the direction that is best for it, and the Property that describes it."""
    formula = prop.get_raw_formula().clone()
    text = str(formula)
    reward = formula.is_reward_operator_formula
    if formula.has_bound:
        compare = COMPARISONS[formula.comparison_type]
        threshold = Fraction(str(formula.threshold_expr.evaluate_as_rational()))
        formula.remove_bound()
        maximizing = compare in (operator.gt, operator.ge)
        direction = stormpy.logic.OptimizationDirection.MAXIMIZE if maximizing else stormpy.logic.OptimizationDirection.MINIMIZE
        formula.set_optimality_type(direction)
        return formula, Property(text, reward, maximizing, compare, threshold)
    maximizing = formula.has_optimality_type and formula.optimality_type == stormpy.logic.OptimizationDirection.MAXIMIZE
    return formula, Property(text, reward, maximizing)


def _value(model, formula):
    result = stormpy.model_checking(model, formula, only_initial_states=True)
    value = result.get_value(model.initial_states[0])
    return value if value == math.inf else Fraction(str(value))


def _valuations(model):
    """The valuations of the states that have no command (Storm gives them a self-loop and the label "deadlock"), and of the others."""
    deadlocks = set(model.labeling.get_states("deadlock"))
    valuations = [str(model.state_valuations.to_string(state)) for state in range(model.nr_states)]
    return frozenset(valuations[state] for state in deadlocks), frozenset(valuations[state] for state in range(model.nr_states) if state not in deadlocks)


def read_properties(props_path):
    """The properties of a properties file, one per line."""
    with open(props_path) as file:
        lines = [line.split("//")[0].strip().rstrip(";") for line in file]
    return [line for line in lines if line]


class Reference:
    def __init__(self, sketch_path, properties):
        """:param properties: the properties, as text"""
        with open(sketch_path) as file:
            lines = file.readlines()
        holes = [(index, HOLE.match(line)) for index, line in enumerate(lines)]
        holes = [(index, match.group(1), match.group(2), hole_options(match.group(3))) for index, match in holes if match]
        self.properties = None
        self.designs = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "design.prism")
            for combination in itertools.product(*[options for _, _, _, options in holes]):
                design_lines = list(lines)
                for (index, kind, name, options), option in zip(holes, combination, strict=True):
                    low, high = min(options, key=number), max(options, key=number)
                    design_lines[index] = f"const {kind} {name} = {option}; const {kind} {name}_MIN = {low}; const {kind} {name}_MAX = {high};\n"
                with open(path, "w") as file:
                    file.writelines(design_lines)
                program = stormpy.parse_prism_program(path, prism_compatibility=True)
                parsed = [stormpy.parse_properties(line, program)[0] for line in properties]
                formulas, self.properties = zip(*[_formula(prop) for prop in parsed], strict=True)
                builder_options = stormpy.BuilderOptions([prop.get_raw_formula() for prop in parsed])
                builder_options.set_build_state_valuations(True)
                model = stormpy.build_model[stormpy.Rational](program, options=builder_options)
                values = tuple(_value(model, formula) for formula in formulas)
                options = tuple((name, option) for (_, _, name, _), option in zip(holes, combination, strict=True))
                self.designs.append(Design(options, values, *_valuations(model)))
        live = frozenset().union(*(design.live for design in self.designs))
        self.designs = [design for design in self.designs if not design.deadlocks & live]

    @property
    def objective(self):
        """The index of the objective among the properties, or None."""
        return next((index for index, prop in enumerate(self.properties) if prop.is_objective), None)

    def admissible(self, design):
        """The design satisfies every constraint, and its value of the objective is valid."""
        return all(prop.valid(value) if prop.is_objective else prop.holds(value) for prop, value in zip(self.properties, design.values, strict=True))

    @functools.cached_property
    def feasible(self):
        return [design for design in self.designs if self.admissible(design)]

    @functools.cached_property
    def optimum(self):
        """The best value of the objective among the admissible designs; None without an objective or an admissible design."""
        if self.objective is None or not self.feasible:
            return None
        best = max if self.properties[self.objective].maximizing else min
        return best(design.values[self.objective] for design in self.feasible)

    def design(self, assignment):
        """The design of a full assignment of PAYNT's parameter space, or None if it is skipped."""
        options = {
            assignment.parameter_name(parameter): number(assignment.parameter_to_option_labels[parameter][assignment.parameter_options(parameter)[0]])
            for parameter in range(assignment.num_parameters)
        }
        return next((design for design in self.designs if all(options[hole] == number(option) for hole, option in design.options)), None)
