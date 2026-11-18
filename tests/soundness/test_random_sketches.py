"""Every engine against the reference on small random sketches: a DTMC of a few states, where a hole decides which distribution some states take, and may leave
one without a command (the designs that reach it there are skipped).

A threshold lies between two values of the designs, at least MARGIN away from all of them, so that the verdict does not depend on the precision.
PAYNT_RANDOM_SKETCHES sets the number of sketches, PAYNT_RANDOM_FIRST_SEED the seed of the first one.
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
    """The text of a random DTMC sketch with an absorbing goal, maybe an absorbing sink and states that some options of a hole leave without a command, a state
    to avoid (labels "avoid" and "safe") and one or two states of a condition."""
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
            if rng.random() < 0.15:
                guards.pop(rng.randrange(options))
        else:
            guards = [(f"s={state}", f"a{state}")]
        for guard, action in guards:
            successors = rng.sample(range(num_states), rng.randint(1, 3))
            weights = [rng.randint(1, 4) for _ in successors]
            distribution = " + ".join(f"{weight}/{sum(weights)} : (s'={successor})" for successor, weight in zip(successors, weights, strict=True))
            lines.append(f"    [{action}] {guard} -> {distribution};")
            rewards.append(f"    [{action}] true : {rng.randint(0, 3)};")
    avoid = rng.randrange(goal)
    condition = " | ".join(f"s={state}" for state in rng.sample(range(num_states), rng.randint(1, 2)))
    labels = [f'label "goal" = s={goal};', f'label "avoid" = s={avoid};', f'label "safe" = s!={avoid};', f'label "condition" = {condition};']
    lines += ["endmodule", "", *labels, "", 'rewards "cost"', *rewards, "endrewards", ""]
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
    """Properties on the goal: one or two constraints, an objective, or an objective with a constraint; a probability may be of reaching the goal without the
    state to avoid, or of reaching it given that the condition is reached."""
    path = rng.choice(['F "goal"', '!"avoid" U "goal"', '"safe" U "goal"', 'F "goal" || F "condition"'])
    # every design may be skipped, or never reach the condition
    probabilities = [design.values[0] for design in Reference(sketch_path, [f"P=? [{path}]"]).designs if design.values[0] is not None] or [Fraction(0)]
    costs = [design.values[0] for design in Reference(sketch_path, ['R{"cost"}=? [F "goal"]']).designs]
    finite_costs = [cost for cost in costs if cost != float("inf")] or [Fraction(0)]
    probability = f"P{rng.choice(['>=', '>', '<=', '<'])}{as_decimal(threshold(rng, probabilities))} [{path}]"
    cost = f'R{{"cost"}}{rng.choice([">=", ">", "<=", "<"])}{as_decimal(threshold(rng, finite_costs))} [F "goal"]'
    objective = rng.choice([f"Pmax=? [{path}]", f"Pmin=? [{path}]", 'R{"cost"}max=? [F "goal"]', 'R{"cost"}min=? [F "goal"]'])
    kind = rng.choice(["probability", "cost", "both", "objective", "constrained objective"])
    return {
        "probability": [probability],
        "cost": [cost],
        "both": [probability, cost],
        "objective": [objective],
        "constrained objective": [rng.choice([probability, cost]), objective],
    }[kind]


@pytest.mark.parametrize("seed", range(FIRST_SEED, FIRST_SEED + SKETCHES))
def test_every_engine_gives_the_reference_answer(seed, tmp_path):
    rng = random.Random(seed)
    sketch_path = tmp_path / "sketch.templ"
    sketch_path.write_text(random_sketch(rng))
    properties = random_specification(rng, sketch_path)
    wrong = wrong_answers(sketch_path, properties, tmp_path)
    assert not wrong, f"{wrong}\n{properties}\n{sketch_path.read_text()}"
