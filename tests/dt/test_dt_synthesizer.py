import pytest
import stormpy

import paynt.dt

from tests.helpers.models import get_sketch_paths, load_sketch
from tests.helpers.reference import read_properties


class TestDtSynthesis:
    def test_synthesize_finds_the_known_optimum(self, dt_orchard):
        """At depth 0 a tree is a single leaf: the best one is `roll`."""
        synthesizer = paynt.dt.DtSynthesizer(dt_orchard.factory, dt_orchard.task)
        synthesizer.synthesize_tree(depth=0)
        assert synthesizer.best_tree is not None
        # SMPMC finds the same, see test_smpmc_inner_engine.py
        assert synthesizer.best_tree_value == pytest.approx(0.48450450140758644, abs=1e-6)
        assert synthesizer.best_tree.to_string() == "roll\n"


class TestTwoConstraints:
    """The trips of models/tests/time-energy as trees: with tree.props the two stages need different speeds."""

    def synthesize(self, depth):
        factory, task = load_sketch("tests/time-energy", sketch_name="tree.templ", props_name="tree.props")
        synthesizer = paynt.dt.DtSynthesizer(factory, task)
        synthesizer.synthesize_tree(depth)
        return synthesizer.best_tree

    def test_no_tree_of_depth_0_satisfies_both(self):
        assert self.synthesize(0) is None

    def test_a_tree_of_depth_1_drives_one_stage_fast(self):
        tree = self.synthesize(1)
        leaves = sorted(tree.action_labels[leaf.action] for leaf in tree.collect_terminals())
        assert leaves in (["fast", "medium"], ["__random__", "fast"])

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="AR drops the root: the tree best for the energy alone drives slowly, which takes 6 hours")
    def test_the_least_energy_of_a_trip_of_4_hours_is_4(self):
        """optimality.props: at most 4 hours, the least energy, at depth 0."""
        factory, task = load_sketch("tests/time-energy", sketch_name="tree.templ", props_name="optimality.props")
        synthesizer = paynt.dt.DtSynthesizer(factory, task)
        synthesizer.synthesize_tree(0)
        assert synthesizer.best_tree_value == pytest.approx(4.0)


def tree_scheduler_value(tmp_path, project, depth, sketch_name="sketch.templ", props_name="sketch.props", **task_kwargs):
    """(the value of the tree, the value of the model with the tree's to_prism module added to the sketch, which synchronizes on the actions)."""
    factory, task = load_sketch(project, sketch_name=sketch_name, props_name=props_name, task_kwargs=task_kwargs)
    synthesizer = paynt.dt.DtSynthesizer(factory, task)
    synthesizer.synthesize_tree(depth)
    sketch_path, props_path = get_sketch_paths(project, sketch_name, props_name)
    with open(sketch_path) as file:
        (tmp_path / "scheduled.prism").write_text(file.read() + "\n" + synthesizer.best_tree.to_prism())
    program = stormpy.parse_prism_program(str(tmp_path / "scheduled.prism"), prism_compatibility=True)
    prop = stormpy.parse_properties(read_properties(props_path)[0], program)[0]
    model = stormpy.build_model(program, [prop])
    return synthesizer.best_tree_value, stormpy.model_checking(model, prop).get_value(model.initial_states[0])


class TestExportToPrism:
    def test_the_scheduler_of_a_tree_has_its_value(self, tmp_path):
        tree_value, scheduler_value = tree_scheduler_value(tmp_path, "tests/dt-maze", depth=2)
        assert scheduler_value == pytest.approx(tree_value, rel=1e-4)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="the guards of the tree block place in the initial state, which the tree does not decide")
    def test_the_scheduler_leaves_the_actions_of_other_states_alone(self, tmp_path):
        tree_value, scheduler_value = tree_scheduler_value(tmp_path, "tests/dt-maze", depth=1)
        assert scheduler_value == pytest.approx(tree_value, rel=1e-4)

    @pytest.mark.xfail(strict=True, raises=RuntimeError, reason="s takes the values 0 and 1, so it is written as a boolean: s, !(s)")
    def test_an_integer_with_the_values_0_and_1_stays_an_integer(self, tmp_path):
        """The trip of time-energy without the don't-care action: at depth 1, one stage fast and the other at medium speed take 3 hours."""
        trip = {"sketch_name": "tree.templ", "props_name": "tree.props", "add_dont_care_action": False}
        _tree_value, time = tree_scheduler_value(tmp_path, "tests/time-energy", depth=1, **trip)
        assert time == pytest.approx(3)


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="the choices of the states that no tree decides are dropped, so the initial state has none")
def test_a_scheduler_from_a_file_is_mapped_to_a_tree(tmp_path):
    """The optimal scheduler of dt-maze, written by Storm, given as --tree-map-scheduler."""
    factory, task = load_sketch("tests/dt-maze")
    mdp = factory.build().underlying_mdp
    scheduler = stormpy.model_checking(mdp, task.get_property().formula, extract_scheduler=True).get_scheduler()
    (tmp_path / "scheduler.json").write_text(scheduler.to_json_str(mdp))
    factory, task = load_sketch("tests/dt-maze", task_kwargs={"scheduler_path": str(tmp_path / "scheduler.json")})
    assert paynt.dt.DtSynthesizer(factory, task).run().success
