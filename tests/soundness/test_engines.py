"""Every engine against the reference answers of tests/helpers/reference.py on tiny sketches: the verdict, the design returned, the value reported and, with a
relative error, how far that value may be from the optimum.

A case runs every engine, on the standard coloring and, for those that support it, on the same coloring lifted into a general one. An engine that does not
support the specification must refuse it with NotImplementedError. Known wrong answers are listed with their case: the case is an expected failure as long as
exactly those engines give exactly those answers, and fails when one of them changes, a fix included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import paynt.parser.sketch
import paynt.synthesizer.search_node
import paynt.synthesizer.statistic
import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_ar
import paynt.synthesizer.synthesizer_hybrid
import paynt.synthesizer.synthesizer_onebyone

from tests.helpers.engines import wrong_answers
from tests.helpers.models import general_colored_mdp, get_sketch_paths, load_sketch
from tests.helpers.reference import read_properties


@dataclass(frozen=True)
class Case:
    sketch_path: str
    properties: tuple
    relative_error: float = 0
    # engine -> its known wrong answer
    known_wrong: dict = field(default_factory=dict)


def case(project, props, sketch_name="sketch.templ", relative_error=0, known_wrong=None, id=None):
    """:param props: a properties file of the project, or the properties themselves"""
    sketch_path = get_sketch_paths(project, sketch_name)[0]
    properties = tuple(read_properties(get_sketch_paths(project, props_name=props)[1]) if isinstance(props, str) else props)
    if id is None:
        id = f"{project.removeprefix('tests/')}/{props}" if isinstance(props, str) else f"{sketch_name.removesuffix('.templ')}: {'; '.join(props)}"
    return pytest.param(Case(sketch_path, properties, relative_error, known_wrong or {}), id=id)


def edge_case(sketch_name, *properties, **kwargs):
    return case("tests/edge-cases", list(properties), sketch_name=sketch_name, **kwargs)


def conditional_case(*properties, sketch_name="undefined.templ", **kwargs):
    return case("tests/conditional", list(properties), sketch_name=sketch_name, **kwargs)


CASES = [
    case("tests/smpmc-tiny", "sketch.props"),
    case("tests/smpmc-tiny", "optimality.props"),
    case("tests/time-energy", "sketch.props"),
    case("tests/time-energy", "infeasible.props"),
    case("tests/time-energy", "optimality.props"),
    case("tests/time-energy", ['R{"energy"}min=? [F "arrived"]'], relative_error=0.5, id="time-energy: least energy, relative error 0.5"),
    case("tests/time-energy", ['R{"time"}max=? [F "arrived"]'], relative_error=0.5, id="time-energy: most time, relative error 0.5"),
    case("tests/objective-optimum-violates-constraint", "sketch.props"),
    case("tests/max-reward-infinite-designs", "sketch.props"),
    case("tests/max-reward-infinite-designs", "lower.props"),
    # families: a design is an MDP
    case("tests/mdp-family-correlated", "sketch.props"),
    case("tests/mdp-family-correlated", "min.props"),
    # one design decides a constraint at the root, but not for every design
    case("tests/constraint-decided-by-one-design", "sketch.props"),
    # thresholds equal to a value of a design
    edge_case("boundaries.templ", 'P>=1 [F "goal"]'),
    edge_case("boundaries.templ", 'P>1 [F "goal"]'),
    edge_case("boundaries.templ", 'P<=0 [F "goal"]'),
    edge_case("boundaries.templ", 'P<0.25 [F "goal"]'),
    edge_case("boundaries.templ", 'P>=0.5 [F "goal"]', 'R{"cost"}<=2 [F "done"]'),
    edge_case("boundaries.templ", 'P>0.5 [F "goal"]', 'R{"cost"}<=2 [F "done"]'),
    edge_case("boundaries.templ", 'R{"cost"}<=0 [F "done"]'),
    edge_case("boundaries.templ", 'R{"cost"}>=4 [F "done"]'),
    edge_case("boundaries.templ", 'Pmin=? [F "goal"]'),
    edge_case("boundaries.templ", 'R{"cost"}max=? [F "done"]'),
    edge_case("boundaries.templ", 'P>=0.5 [F "goal"]', 'R{"cost"}min=? [F "done"]'),
    edge_case("boundaries.templ", 'P<=0.5 [F "goal"]', 'Pmax=? [F "goal"]'),
    edge_case("boundaries.templ", 'P>=0.5 [!"done" U "goal"]'),
    # a goal reached at once, and one never reached
    edge_case("reach.templ", 'P>=1 [F "start"]'),
    edge_case("reach.templ", 'R{"steps"}max=? [F "start"]'),
    edge_case("reach.templ", 'P>0 [F "nowhere"]'),
    edge_case("reach.templ", 'Pmax=? [F "nowhere"]'),
    edge_case("reach.templ", 'R{"steps"}min=? [F "nowhere"]'),
    # degenerate parameters
    edge_case("one-design.templ", 'P>0.5 [F "goal"]'),
    edge_case("one-design.templ", 'Pmax=? [F "goal"]'),
    edge_case("unused-hole.templ", 'P>=0.5 [F "goal"]'),
    edge_case("unused-hole.templ", 'Pmin=? [F "goal"]'),
    edge_case("probability-hole.templ", 'P>=0.5 [F "goal"]'),
    edge_case("probability-hole.templ", 'Pmax=? [F "goal"]'),
    # an until formula: checked as it is if a state where neither side holds can reach the goal, as reaching the goal otherwise
    edge_case("until.templ", 'Pmax=? [!"bad" U "goal"]'),
    edge_case("until.templ", 'Pmin=? [!"bad" U "goal"]'),
    edge_case("until.templ", 'P>=0.4 [!"bad" U "goal"]'),
    edge_case("until.templ", 'P<=0.4 [!"bad" U "goal"]'),
    edge_case("until.templ", 'Pmax=? ["ok" U "goal"]'),
    edge_case("until.templ", 'P>=0.1 [F "goal"]', 'Pmax=? ["ok" U "goal"]'),
    edge_case("until.templ", 'P>=0.4 ["ok" U "goal"]', 'P>=0.4 [F "goal"]'),
    # Storm's scheduler for the least cost is not a design of least cost
    edge_case("zero-cost-loop.templ", 'R{"cost"}min=? [F "goal"]'),
    edge_case("zero-cost-loop.templ", 'P>=0.5 [F "goal"]', 'R{"cost"}min=? [F "goal"]'),
    # designs that reach a state without a command are skipped
    edge_case("deadlock.templ", 'Pmin=? [F "goal"]'),
    edge_case("deadlock-unreached-command.templ", 'P>=0.25 [F "goal"]'),
    edge_case("deadlock-unreached-command.templ", 'P<0.25 [F "goal"]'),
    edge_case("deadlock-unreached-command.templ", 'Pmax=? [F "goal"]'),
    edge_case("deadlock-family.templ", 'P>=0.4 [F "goal"]'),
    edge_case("deadlock-family.templ", 'Pmin=? [F "goal"]'),
    edge_case("deadlock-family.templ", 'R{"cost"}min=? [F "done"]'),
    edge_case("deadlock-family.templ", 'R{"cost"}max=? [F "done"]'),
    edge_case("deadlock-family.templ", 'P>=0.4 [F "goal"]', 'R{"cost"}<=1.5 [F "done"]'),
    # conditional properties: a policy with memory does better than any design, and a design that never reaches the condition satisfies nothing
    case("tests/conditional", "sketch.props"),
    case("tests/conditional", ['P>=0.5 [F "goal" || F "condition"]']),
    case("tests/conditional", ['P>0.5 [F "goal" || F "condition"]']),
    conditional_case('Pmin=? [F "goal" || F "condition"]'),
    conditional_case('Pmax=? [F "goal" || F "condition"]'),
    conditional_case('P<=0.8 [F "goal" || F "condition"]'),
    conditional_case('P>=0.5 [F "goal" || F "condition"]'),
    conditional_case('P<0.25 [F "goal" || F "condition"]'),
    conditional_case('P>=0.25 [F "goal" || F "condition"]', 'R{"steps"}min=? [F "goal"]'),
    # around Storm's problems with conditional properties
    conditional_case('Pmin=? [F "goal" || F "condition"]', sketch_name="goal-first.templ"),
    conditional_case('Pmin=? [F "goal" || F "condition"]', sketch_name="almost-sure-condition.templ"),
    conditional_case('P<=0.75 [F "goal" || F "condition"]', sketch_name="almost-sure-condition.templ"),
    conditional_case('P>=-0.1 [F "goal" || F "condition"]', sketch_name="threshold.templ"),
    conditional_case('P>=-0.1 [F "goal" || F "condition"]', 'Pmin=? [F "goal" || F "condition"]', sketch_name="threshold.templ"),
]


@pytest.mark.parametrize("case", CASES)
def test_every_engine_gives_the_reference_answer(case, tmp_path):
    wrong = wrong_answers(case.sketch_path, case.properties, tmp_path, case.relative_error)
    # a known wrong answer is given by its beginning
    assert {engine: problem[: len(case.known_wrong.get(engine, problem))] for engine, problem in wrong.items()} == case.known_wrong
    if wrong:
        pytest.xfail("known wrong answers")


@pytest.mark.parametrize("engine", ["onebyone", "cegis", "smpmc", "onebyone, general", "cegis, general", "smpmc, general"])
def test_an_engine_that_tries_a_design_with_a_deadlock_skips_it_and_warns(engine, tmp_path, caplog):
    """In deadlock.templ, H=1 enables no command in state 1; these engines try it, as H=0 alone does not show it worse."""
    method, _, coloring = engine.partition(", ")
    props_path = tmp_path / "case.props"
    props_path.write_text('Pmin=? [F "goal"]\n')
    factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths("tests/edge-cases", "deadlock.templ")[0], str(props_path))
    colored_mdp = factory.build() if coloring != "general" else general_colored_mdp(factory.build())
    result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
    assert str(result.assignment) == "H=0"
    assert [record.getMessage() for record in caplog.records if record.levelname == "WARNING"] == [
        "skipping design H=1: state 1 (goal=False, s=1) is reachable and keeps no choice"
    ]


class TestAConstraintDecidedByOneDesign:
    """The sketch hybrid.templ of models/tests/constraint-decided-by-one-design: only H=1 satisfies both constraints, H=2 reaches the goal half of the time."""

    def test_ar_leaves_the_cost_constraint_open_at_the_root(self):
        factory, task = load_sketch("tests/constraint-decided-by-one-design", sketch_name="hybrid.templ", props_name="hybrid.props")
        synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(factory.build(), task)
        synthesizer.stat = paynt.synthesizer.statistic.Statistic(synthesizer)
        root = paynt.synthesizer.search_node.SearchNode(synthesizer.colored_mdp.parameter_space)
        synthesizer.verify_parameter_space(root)
        assert [result.sat for result in root.analysis_result.constraints_result.results] == [None, None]

    def test_hybrid_without_cegis_finds_h_1(self, monkeypatch):
        """Without CEGIS, which may find H=1 at the root first, hybrid searches like AR."""
        monkeypatch.setattr(paynt.synthesizer.synthesizer_hybrid.StageControl, "only_ar", True)
        factory, task = load_sketch("tests/constraint-decided-by-one-design", sketch_name="hybrid.templ", props_name="hybrid.props")
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, "hybrid").run()
        assert str(result.assignment) == "H=1"
