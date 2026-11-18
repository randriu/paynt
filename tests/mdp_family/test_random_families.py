"""The policy tree against the reference on small random MDP families: a hole decides the outcome of some actions, and whether some are offered at all.

Every state keeps an action that no hole decides, so no member is left without a choice. The members the policy tree satisfies must be those the reference finds
feasible, and each of them must satisfy the property with the policy it is given, offering every action that policy takes. PAYNT_RANDOM_FAMILIES sets the number
of families, PAYNT_RANDOM_FIRST_SEED the seed of the first one.
"""

from __future__ import annotations

import os
import random

import pytest

import paynt.mdp_family
import paynt.mdp_family._utils
import paynt.model.model
import paynt.parser.sketch
import paynt.utils.error_handling

from tests.helpers.reference import Reference
from tests.soundness.test_random_sketches import as_decimal, threshold

FAMILIES = int(os.environ.get("PAYNT_RANDOM_FAMILIES", "25"))
FIRST_SEED = int(os.environ.get("PAYNT_RANDOM_FIRST_SEED", "0"))


def distribution(rng, num_states):
    successors = rng.sample(range(num_states), rng.randint(1, 3))
    weights = [rng.randint(1, 4) for _ in successors]
    return " + ".join(f"{weight}/{sum(weights)} : (s'={successor})" for successor, weight in zip(successors, weights, strict=True))


def random_family(rng):
    """The text of a random MDP family with an absorbing goal and maybe an absorbing sink."""
    num_states = rng.randint(3, 5)
    goal = num_states - 1
    absorbing = {goal, goal - 1} if rng.random() < 0.5 else {goal}
    holes = [(f"H{index}", rng.randint(2, 3)) for index in range(rng.randint(1, 2))]
    lines = ["mdp", ""]
    lines += [f"hole int {name} in {{0..{options - 1}}};" for name, options in holes]
    lines += ["", "module m", f"    s : [0..{goal}] init 0;"]
    for state in range(num_states):
        if state in absorbing:
            lines.append(f"    [] s={state} -> true;")
            continue
        # the first action is decided by no hole; the others may depend on one, and some of its options may not offer them
        for index, action in enumerate(rng.sample(["a", "b", "c"], rng.randint(1, 3))):
            if index == 0 or rng.random() < 0.3:
                lines.append(f"    [{action}{state}] s={state} -> {distribution(rng, num_states)};")
                continue
            name, options = rng.choice(holes)
            offered = [option for option in range(options) if index == 0 or rng.random() < 0.8] or [0]
            for option in offered:
                lines.append(f"    [{action}{state}] s={state} & {name}={option} -> {distribution(rng, num_states)};")
    lines += ["endmodule", "", f'label "goal" = s={goal};', ""]
    return "\n".join(lines)


def satisfies(colored_mdp, prop, member, policy):
    """Whether the member offers every action the policy takes where it goes, and satisfies prop with it."""
    choices = colored_mdp.coloring.selectCompatibleChoices(member.native)
    try:
        _, mdp = paynt.mdp_family._utils.fix_and_apply_policy_to_parameter_space(colored_mdp, choices, policy)
    except paynt.utils.error_handling.StateWithoutChoice:
        # the policy takes an action that the member does not offer
        return False
    return mdp.model_check_property(prop, alt=True).sat


@pytest.mark.parametrize("seed", range(FIRST_SEED, FIRST_SEED + FAMILIES))
def test_the_policy_tree_satisfies_the_feasible_members(seed, tmp_path):
    rng = random.Random(seed)
    sketch_path = tmp_path / "sketch.templ"
    sketch_path.write_text(random_family(rng))
    best = [design.values[0] for design in Reference(sketch_path, ['Pmax=? [F "goal"]']).designs]
    worst = [design.values[0] for design in Reference(sketch_path, ['Pmin=? [F "goal"]']).designs]
    direction = rng.choice([">=", "<="])
    prop_text = f'P{direction}{as_decimal(threshold(rng, best if direction == ">=" else worst))} [F "goal"]'
    props_path = tmp_path / "sketch.props"
    props_path.write_text(prop_text + "\n")

    factory, task = paynt.parser.sketch.Sketch.load_sketch(str(sketch_path), str(props_path))
    colored_mdp = factory.build()
    synthesizer = paynt.mdp_family.PolicyTreeSynthesizer(colored_mdp, task)
    prop = task.get_property()
    satisfied = set()
    for evaluation in synthesizer.evaluate(prop=prop, print_stats=False):
        if not evaluation.sat:
            continue
        space = evaluation.parameter_space
        for combination in space.all_combinations():
            member = space.construct_assignment(combination)
            assert satisfies(colored_mdp, prop, member, evaluation.policy[0]), f"{member} with its policy\n{prop_text}\n{sketch_path.read_text()}"
            satisfied.add(str(member))
    feasible = {str(design) for design in Reference(sketch_path, [prop_text]).feasible}
    assert satisfied == feasible, f"{prop_text}\n{sketch_path.read_text()}"
