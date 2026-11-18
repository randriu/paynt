"""SynthesizerOneByOne.evaluate: the value of every design."""

import json

import pytest

import paynt.synthesizer.synthesizer_onebyone

from tests.helpers.models import get_sketch_paths, load_colored_mdp
from tests.helpers.reference import Reference


def test_onebyone_evaluates_every_design_to_its_reference_value():
    """The energy of every trip of models/tests/time-energy."""
    colored_mdp, task = load_colored_mdp("tests/time-energy", props_name="optimality.props")
    synthesizer = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(colored_mdp, task)
    values = synthesizer.evaluate(prop=task.specification.optimality, keep_value_only=True, print_stats=False)
    reference = Reference(get_sketch_paths("tests/time-energy")[0], ['R{"energy"}=? [F "arrived"]'])
    space = colored_mdp.parameter_space
    designs = [reference.design(space.construct_assignment(combination)) for combination in space.all_combinations()]
    assert values == pytest.approx([float(design.values[0]) for design in designs])


def test_the_evaluation_is_exported_member_by_member(tmp_path):
    colored_mdp, task = load_colored_mdp("tests/mdp-family-correlated")
    task.export_synthesis_filename_base = str(tmp_path / "evaluation")
    paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(colored_mdp, task).evaluate(print_stats=False)
    members = json.loads((tmp_path / "evaluation.json").read_text())
    # the best policy of each member, 0.54 both (worked out in the sketch)
    assert members == [["h=0", [[{"s": 0}, "a"], [{"s": 1}, "d"]]], ["h=1", [[{"s": 0}, "b"], [{"s": 1}, "c"]]]]
