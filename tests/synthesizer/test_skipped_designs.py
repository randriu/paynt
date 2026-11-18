"""Designs that reach a state without a choice, on models/tests/edge-cases/deadlock-family.templ: G=0 with H=1 or H=2 is skipped, whatever K."""

import json
import random

import pytest

import paynt.parser.sketch
import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc._utils
import paynt.synthesizer.smpmc.checker
import paynt.synthesizer.synthesizer_cegis
import paynt.utils.error_handling

from tests.helpers.models import general_colored_mdp, get_sketch_paths
from tests.soundness.test_random_sketches import random_sketch


@pytest.fixture
def family(tmp_path):
    props_path = tmp_path / "case.props"
    props_path.write_text('Pmin=? [F "goal"]\n')
    factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths("tests/edge-cases", "deadlock-family.templ")[0], str(props_path))
    return factory.build(), task


def space(colored_mdp, **options):
    """The parameter space with the given options of G, H and K, all of them for the others."""
    result = colored_mdp.parameter_space.copy()
    for parameter in range(result.num_parameters):
        name = result.parameter_name(parameter)
        if name in options:
            result.parameter_set_options(parameter, options[name])
    return result


class TestTheSubMdp:
    def test_keeps_the_choices_that_avoid_the_state_without_one(self, family):
        colored_mdp, _task = family
        mdp, _choices = colored_mdp.build(space(colored_mdp, H=[1, 2]))
        # G=0 goes to state 1, which keeps no choice: only G=1 is left in state 0
        valuations = colored_mdp.underlying_mdp.state_valuations
        assert [json.loads(str(valuations.to_json(state)))["s"] for state in mdp.underlying_mdp_state_map] == [0, 3]

    def test_skips_every_design_when_none_avoids_it(self, family):
        colored_mdp, _task = family
        with pytest.raises(paynt.utils.error_handling.DeadlockedDesigns, match="^skipping the 4 designs of"):
            colored_mdp.build(space(colored_mdp, G=[0], H=[1, 2]))


def test_cegis_skips_the_designs_that_agree_on_the_way_to_the_state(family, caplog):
    colored_mdp, task = family
    synthesizer = paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS(colored_mdp, task)
    with pytest.raises(paynt.utils.error_handling.DeadlockedDesigns) as raised:
        colored_mdp.build_assignment(space(colored_mdp, G=[0], H=[1], K=[0]))
    conflict = synthesizer.deadlock_conflict(raised.value)
    assert [colored_mdp.parameter_space.parameter_name(parameter) for parameter in conflict] == ["G", "H"]
    assert caplog.records[-1].getMessage().endswith("is reachable and keeps no choice; so is every design that agrees with it on G, H")


def test_smpmc_keeps_the_parameters_of_the_state_without_a_choice_in_a_conflict(tmp_path):
    """With H=1 the sub-MDP keeps only G=1 in state 0, as G=0 leads to state 1, which keeps no choice: H decides the sub-MDP, though state 1 is not in it."""
    props_path = tmp_path / "case.props"
    props_path.write_text('P>=0.4 [F "goal"]\n')
    factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths("tests/edge-cases", "deadlock-family.templ")[0], str(props_path))
    colored_mdp = factory.build()
    theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, task.get_property())
    [h] = [parameter for parameter in range(colored_mdp.parameter_space.num_parameters) if colored_mdp.parameter_space.parameter_name(parameter) == "H"]
    assert theory.check({h: 1}, True).conflict_parameters == [h]


class TestARobustPolicyKeepsAChoiceInEveryEnvironment:
    """deadlock-environment.templ: the policy P=0 gets stuck in the environment E=1, P=1 reaches the goal with probability 1/2 in every environment."""

    @staticmethod
    def load(tmp_path, prop, verify_robust=True):
        props_path = tmp_path / "case.props"
        props_path.write_text(prop + "\n")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths("tests/edge-cases", "deadlock-environment.templ")[0], str(props_path))
        task.constraint_name = "exists_forall"
        task.forall_pattern = "^E$"
        task.verify_robust = verify_robust
        return factory.build(), task

    def test_a_policy_stuck_in_one_environment_is_not_robust(self, tmp_path):
        colored_mdp, task = self.load(tmp_path, 'P>=0.9 [F "goal"]')
        assert not paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run().success

    def test_the_policy_that_keeps_a_choice_is_robust(self, tmp_path):
        colored_mdp, task = self.load(tmp_path, 'P>=0.4 [F "goal"]')
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run()
        assert str(result.robust_assignment) == "P=1, E: {0,1}"
        assert result.robust_verified

    def test_its_worst_case_is_the_robust_optimum(self, tmp_path):
        colored_mdp, task = self.load(tmp_path, 'Pmax=? [F "goal"]')
        result = paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run()
        assert (str(result.robust_assignment), result.value) == ("P=1, E: {0,1}", pytest.approx(0.5))
        assert result.robust_verified

    def test_the_worst_case_of_the_environments_of_a_policy_does_not_speak_for_one_that_gets_stuck(self, tmp_path):
        """With P=0 the sub-MDP holds E=0's command in state 1, so its worst case is 1."""
        colored_mdp, task = self.load(tmp_path, 'P>=0.9 [F "goal"]')
        p, e = range(2)
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, task.get_property(), forall_parameters=[e])
        assert theory.check({p: 0}, False) is None

    def test_the_verification_finds_the_environment_where_a_policy_gets_stuck(self, tmp_path):
        colored_mdp, task = self.load(tmp_path, 'P>=0.4 [F "goal"]')
        assert str(colored_mdp.find_deadlocked_design(space(colored_mdp, P=[0]))) == "P=0, E=1"
        assert colored_mdp.find_deadlocked_design(space(colored_mdp, P=[1])) is None
        assert not paynt.synthesizer.smpmc._utils.verify_robust(colored_mdp, task, space(colored_mdp, P=[0]))


@pytest.mark.parametrize("coloring", ["standard", "general"])
def test_the_states_some_designs_leave_without_a_choice(family, coloring):
    """Only H=0 has a command in state s=1."""
    colored_mdp = family[0] if coloring == "standard" else general_colored_mdp(family[0])
    valuations = colored_mdp.underlying_mdp.state_valuations
    states = range(colored_mdp.underlying_mdp.nr_states)
    uncovered = colored_mdp.states_without_a_choice(colored_mdp.parameter_space, states)
    assert [json.loads(str(valuations.to_json(state)))["s"] for state in uncovered] == [1]
    assert colored_mdp.states_without_a_choice(space(colored_mdp, H=[0]), states) == []


@pytest.mark.parametrize("seed", range(30))
def test_both_colorings_find_the_same_states_without_a_choice(seed, tmp_path):
    """On a random sketch, for the whole space and for each option of each hole: the standard coloring splits over the options, the general one asks Z3."""
    rng = random.Random(seed)
    sketch_path = tmp_path / "sketch.templ"
    sketch_path.write_text(random_sketch(rng))
    props_path = tmp_path / "case.props"
    props_path.write_text('Pmax=? [F "goal"]\n')
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(str(sketch_path), str(props_path))
    standard = factory.build()
    general = general_colored_mdp(standard)
    full = standard.parameter_space
    spaces = [full] + [
        full.assume_parameter_options_copy(parameter, [option]) for parameter in range(full.num_parameters) for option in full.parameter_options(parameter)
    ]
    states = range(standard.underlying_mdp.nr_states)
    for parameter_space in spaces:
        assert standard.states_without_a_choice(parameter_space, states) == general.states_without_a_choice(parameter_space, states)
