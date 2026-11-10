"""Every engine against the reference on small random sketches: a DTMC of a few states, where a hole decides which distribution some states take.

A threshold lies between two values of the designs, at least MARGIN away from all of them, so that the verdict does not depend on the precision. No
specification maximizes a reward or has an objective together with a constraint, which hit known bugs (see test_engines.py). Two constraints, and costs some
design never finishes paying, still hit known bugs of AR in about 2 of 1000 sketches, though not in the default 25. PAYNT_RANDOM_SKETCHES sets the number of
sketches, PAYNT_RANDOM_FIRST_SEED the seed of the first one.
"""

from __future__ import annotations

import itertools
import os
import random
from fractions import Fraction

import pytest

from tests.helpers.engines import wrong_answers
from tests.helpers.reference import Reference

MARGIN = Fraction(1, 1000)
SKETCHES = int(os.environ.get("PAYNT_RANDOM_SKETCHES", "25"))
FIRST_SEED = int(os.environ.get("PAYNT_RANDOM_FIRST_SEED", "0"))


def random_sketch(rng):
    """The text of a random DTMC sketch with an absorbing goal, and maybe an absorbing sink."""
    num_states = rng.randint(3, 6)
    goal = num_states - 1
    absorbing = {goal, goal - 1} if rng.random() < 0.5 else {goal}
    holes = [(f"H{index}", rng.randint(2, 3)) for index in range(rng.randint(1, 3))]
    lines = ["dtmc", ""]
    lines += [f"hole int {name} in {{0..{options - 1}}};" for name, options in holes]
    lines += ["", "module m", f"    s : [0..{goal}] init 0;"]
    rewards = []
    for state in range(num_states):
        if state in absorbing:
            lines.append(f"    [] s={state} -> true;")
            continue
        if rng.random() < 0.7:
            name, options = rng.choice(holes)
            guards = [(f"s={state} & {name}={option}", f"a{state}_{option}") for option in range(options)]
        else:
            guards = [(f"s={state}", f"a{state}")]
        for guard, action in guards:
            successors = rng.sample(range(num_states), rng.randint(1, 3))
            weights = [rng.randint(1, 4) for _ in successors]
            distribution = " + ".join(f"{weight}/{sum(weights)} : (s'={successor})" for successor, weight in zip(successors, weights, strict=True))
            lines.append(f"    [{action}] {guard} -> {distribution};")
            rewards.append(f"    [{action}] true : {rng.randint(0, 3)};")
    lines += ["endmodule", "", f'label "goal" = s={goal};', "", 'rewards "cost"', *rewards, "endrewards", ""]
    return "\n".join(lines)


def threshold(rng, values):
    """A number between two of the values, or beyond all of them, at least MARGIN away from each."""
    values = sorted(set(values))
    gaps = [(low + high) / 2 for low, high in itertools.pairwise(values) if high - low >= 2 * MARGIN]
    return rng.choice(gaps + [values[0] - Fraction(1, 10), values[-1] + Fraction(1, 10)])


def as_decimal(number):
    """A threshold as PRISM reads it, rounded to a value at least MARGIN / 2 away from where it was."""
    return f"{float(number):.6f}"


def random_specification(rng, sketch_path):
    """Properties on the goal: one or two constraints, or one objective."""
    probabilities = [design.values[0] for design in Reference(sketch_path, ['P=? [F "goal"]']).designs]
    costs = [design.values[0] for design in Reference(sketch_path, ['R{"cost"}=? [F "goal"]']).designs]
    finite_costs = [cost for cost in costs if cost != float("inf")] or [Fraction(0)]
    probability = f'P{rng.choice([">=", ">", "<=", "<"])}{as_decimal(threshold(rng, probabilities))} [F "goal"]'
    cost = f'R{{"cost"}}{rng.choice(["<=", "<"])}{as_decimal(threshold(rng, finite_costs))} [F "goal"]'
    objectives = ['Pmax=? [F "goal"]', 'Pmin=? [F "goal"]', 'R{"cost"}min=? [F "goal"]']
    kind = rng.choice(["probability", "cost", "both", "objective"])
    return {"probability": [probability], "cost": [cost], "both": [probability, cost], "objective": [rng.choice(objectives)]}[kind]


@pytest.mark.parametrize("seed", range(FIRST_SEED, FIRST_SEED + SKETCHES))
def test_every_engine_gives_the_reference_answer(seed, tmp_path):
    rng = random.Random(seed)
    sketch_path = tmp_path / "sketch.templ"
    sketch_path.write_text(random_sketch(rng))
    properties = random_specification(rng, sketch_path)
    wrong = wrong_answers(sketch_path, properties, tmp_path)
    assert not wrong, f"{wrong}\n{properties}\n{sketch_path.read_text()}"
