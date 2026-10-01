"""Where a hole may be in a sketch, and where it may not: the reward of a state and a label are refused.

PAYNT substitutes the options of a hole into the transitions of the model. The reward of a state and a label are not on a transition, so a hole in them would
stay there, undefined, and the reward or the label would come out wrong (zero, or true everywhere) without any error: it is an error instead.
"""

from __future__ import annotations

import textwrap

import pytest

import paynt.parser.sketch
import paynt.synthesizer.synthesizer

SKETCH = textwrap.dedent("""
    {model_type}

    hole int POWER in {{1..3}};
    hole int STAY in {{0..1}};
    {definitions}

    module m
        phase : [0..2] init 0;
        [go] phase=0 -> 0.5 : (phase'=1) + 0.5 : (phase'=0);
        [] phase>=1 -> (phase'=phase);
    endmodule

    label "done" = {label};

    {rewards}
    """)


def load(tmp_path, rewards="", definitions="", label="phase=1", model_type="dtmc", props='R{"cost"}min=? [F "done"]'):
    (tmp_path / "sketch.templ").write_text(SKETCH.format(model_type=model_type, definitions=definitions, label=label, rewards=rewards))
    (tmp_path / "sketch.props").write_text(props + "\n")
    return paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))


def optimum(tmp_path, **kwargs):
    factory, task = load(tmp_path, **kwargs)
    return paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, "onebyone").run()


class TestARewardOfAStateMayNotDependOnAHole:
    @pytest.mark.parametrize(
        ("rewards", "definitions"),
        [
            ('rewards "cost" phase=0 : POWER; endrewards', ""),
            ('rewards "cost" phase=POWER-1 : 1; endrewards', ""),
            ('rewards "cost" phase=0 : doubled; endrewards', "formula doubled = 2 * POWER;"),
            ('rewards "cost" phase=0 : redoubled; endrewards', "formula doubled = 2 * POWER; formula redoubled = doubled + 1;"),
            ('rewards "cost" phase=0 : DOUBLED; endrewards', "const int DOUBLED = 2 * POWER;"),
            ('rewards "cost" phase=0 : twice; endrewards', "const int DOUBLED = 2 * POWER; formula twice = DOUBLED + 1;"),
            ('rewards "cost" phase=0 : 1; phase=1 : POWER; endrewards', ""),
        ],
        ids=["value", "guard", "formula", "formula of a formula", "constant", "formula of a constant", "second item"],
    )
    def test_a_hole_is_refused_whether_it_is_used_directly_or_not(self, tmp_path, rewards, definitions):
        with pytest.raises(ValueError, match=r"the state reward 'cost' depends on the hole POWER, which is not supported"):
            load(tmp_path, rewards=rewards, definitions=definitions)

    def test_every_hole_is_named(self, tmp_path):
        with pytest.raises(ValueError, match=r"depends on the holes POWER, STAY"):
            load(tmp_path, rewards='rewards "cost" phase=STAY : POWER; endrewards')

    def test_the_reward_that_does_is_named_among_those_that_do_not(self, tmp_path):
        rewards = 'rewards "time" phase=0 : 1; endrewards rewards "energy" phase=0 : POWER; endrewards'
        with pytest.raises(ValueError, match=r"the state reward 'energy' depends on the hole POWER"):
            load(tmp_path, rewards=rewards, props='R{"time"}min=? [F "done"]')

    def test_the_message_says_what_to_do(self, tmp_path):
        with pytest.raises(ValueError, match=r"Use the hole in the reward of a transition \(\[action\] guard : value;\) instead"):
            load(tmp_path, rewards='rewards "cost" phase=0 : POWER; endrewards')

    @pytest.mark.parametrize("model_type", ["dtmc", "mdp"])
    def test_the_model_type_does_not_matter(self, tmp_path, model_type):
        with pytest.raises(ValueError, match="depends on the hole POWER"):
            load(tmp_path, rewards='rewards "cost" phase=0 : POWER; endrewards', model_type=model_type)


class TestALabelMayNotDependOnAHole:
    def test_a_hole_in_a_label_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"the label 'done' depends on the hole POWER, which is not supported"):
            load(tmp_path, label="phase=POWER", rewards='rewards "cost" phase=0 : 1; endrewards')

    def test_the_message_says_what_to_do(self, tmp_path):
        with pytest.raises(ValueError, match="Put the condition into the property, or into the guards of the commands, instead"):
            load(tmp_path, label="phase=POWER", rewards='rewards "cost" phase=0 : 1; endrewards')


class TestWhereAHoleIsFine:
    def test_the_reward_of_a_transition_may_depend_on_a_hole(self, tmp_path):
        """Two steps in expectation to get done, each costing POWER: the cheapest hole is POWER = 1."""
        result = optimum(tmp_path, rewards='rewards "cost" [go] true : POWER; endrewards')
        assert result.value == pytest.approx(2.0)
        assert str(result.assignment).startswith("POWER=1")

    def test_so_may_the_guard_of_the_reward_of_a_transition(self, tmp_path):
        """A step costs 3 when POWER > 1 and 5 otherwise: 6 is the cheapest."""
        result = optimum(tmp_path, rewards='rewards "cost" [go] POWER>1 : 3; [go] POWER<=1 : 5; endrewards')
        assert result.value == pytest.approx(6.0)
        assert not str(result.assignment).startswith("POWER=1")

    def test_a_hole_may_be_anywhere_else_while_the_reward_of_a_state_depends_on_none(self, tmp_path):
        """The hole STAY is in an update: with STAY = 0 the sender gets done, with STAY = 1 it never does. The time is a reward of a state, without a hole."""
        sketch = SKETCH.format(model_type="dtmc", definitions="", label="phase=1", rewards='rewards "cost" phase=0 : 1; endrewards')
        sketch = sketch.replace("0.5 : (phase'=1) + 0.5 : (phase'=0)", "0.5 : (phase'=STAY+1) + 0.5 : (phase'=0)")
        (tmp_path / "sketch.templ").write_text(sketch)
        (tmp_path / "sketch.props").write_text('R{"cost"}min=? [F "done"]\n')
        factory, task = paynt.parser.sketch.Sketch.load_sketch(str(tmp_path / "sketch.templ"), str(tmp_path / "sketch.props"))
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, "onebyone").run()
        assert result.value == pytest.approx(2.0)
        assert "STAY=0" in str(result.assignment)
