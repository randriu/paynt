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
from tests.helpers.models import get_sketch_paths, load_sketch
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


# the node is dropped when the design best for the objective alone violates a constraint
CONSTRAINT_AND_OBJECTIVE = {"ar": "says infeasible", "hybrid": "says infeasible"}
# the MDP of a family has an infinite maximal reward when a scheduler never reaches the goal, even if every design does, and that is taken for an invalid
# result
MAXIMIZED_INFINITE_REWARD = {"ar": "says infeasible", "smpmc": "says infeasible", "smpmc, general": "says infeasible"}
# a design with a deadlock stops the search instead of being skipped
DEADLOCK = {
    engine: "raises ValueError: the coloring enables no choice in state 1" if not engine.endswith("general") else "raises RuntimeError: UnexpectedException"
    for engine in ["onebyone", "cegis", "smpmc", "onebyone, general", "cegis, general", "smpmc, general"]
}
# a constraint is taken as satisfied by every design when the scheduler best for it is one design that satisfies the specification
NOTHING_TO_SPLIT_ON = "raises AssertionError: undecided node has neither cross-constraint disagreement nor any inconsistent property to split on"
CONSTRAINT_DECIDED_BY_ONE_DESIGN = {"ar": NOTHING_TO_SPLIT_ON, "hybrid": NOTHING_TO_SPLIT_ON}

CASES = [
    case("tests/smpmc-tiny", "sketch.props"),
    case("tests/smpmc-tiny", "optimality.props"),
    case("tests/time-energy", "sketch.props"),
    case("tests/time-energy", "infeasible.props"),
    case("tests/time-energy", "optimality.props", known_wrong=CONSTRAINT_AND_OBJECTIVE),
    case("tests/time-energy", ['R{"energy"}min=? [F "arrived"]'], relative_error=0.5, id="time-energy: least energy, relative error 0.5"),
    case("tests/time-energy", ['R{"time"}max=? [F "arrived"]'], relative_error=0.5, id="time-energy: most time, relative error 0.5"),
    case("tests/objective-optimum-violates-constraint", "sketch.props", known_wrong=CONSTRAINT_AND_OBJECTIVE),
    case("tests/max-reward-infinite-designs", "sketch.props", known_wrong=MAXIMIZED_INFINITE_REWARD),
    case("tests/max-reward-infinite-designs", "lower.props", known_wrong=MAXIMIZED_INFINITE_REWARD),
    case("tests/constraint-decided-by-one-design", "sketch.props", known_wrong=CONSTRAINT_DECIDED_BY_ONE_DESIGN),
    # families: a design is an MDP
    case("tests/mdp-family-correlated", "sketch.props"),
    case("tests/mdp-family-correlated", "min.props"),
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
    edge_case("boundaries.templ", 'P>=0.5 [F "goal"]', 'R{"cost"}min=? [F "done"]', known_wrong=CONSTRAINT_AND_OBJECTIVE),
    edge_case("boundaries.templ", 'P<=0.5 [F "goal"]', 'Pmax=? [F "goal"]', known_wrong=CONSTRAINT_AND_OBJECTIVE),
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
    # with H=1 no command is enabled in state 1: the design is skipped
    edge_case("deadlock.templ", 'Pmin=? [F "goal"]', known_wrong=DEADLOCK),
]


@pytest.mark.parametrize("case", CASES)
def test_every_engine_gives_the_reference_answer(case, tmp_path):
    wrong = wrong_answers(case.sketch_path, case.properties, tmp_path, case.relative_error)
    # a known wrong answer is given by its beginning
    assert {engine: problem[: len(case.known_wrong.get(engine, problem))] for engine, problem in wrong.items()} == case.known_wrong
    if wrong:
        pytest.xfail("known wrong answers")


@pytest.mark.xfail(strict=True, raises=ValueError, reason="OneByOne stops with an error at H=1")
def test_onebyone_skips_a_design_with_a_deadlock_and_warns(tmp_path, caplog):
    """OneByOne builds every design of deadlock.templ, so it comes across H=1, which enables no command in state 1."""
    props_path = tmp_path / "case.props"
    props_path.write_text('Pmin=? [F "goal"]\n')
    factory, task = paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths("tests/edge-cases", "deadlock.templ")[0], str(props_path))
    result = paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, "onebyone").run()
    assert str(result.assignment) == "H=0"
    assert any(record.levelname == "WARNING" and "H=1" in record.getMessage() for record in caplog.records)


class TestAConstraintDecidedByOneDesign:
    """The sketch hybrid.templ of models/tests/constraint-decided-by-one-design: only H=1 satisfies both constraints, H=2 reaches the goal half of the time."""

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="the cheapest scheduler is H=1 alone, which satisfies the specification")
    def test_ar_leaves_the_cost_constraint_open_at_the_root(self):
        factory, task = load_sketch("tests/constraint-decided-by-one-design", sketch_name="hybrid.templ", props_name="hybrid.props")
        synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(factory.build(), task)
        synthesizer.stat = paynt.synthesizer.statistic.Statistic(synthesizer)
        root = paynt.synthesizer.search_node.SearchNode(synthesizer.colored_mdp.parameter_space)
        synthesizer.verify_parameter_space(root)
        assert [result.sat for result in root.analysis_result.constraints_result.results] == [None, None]

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason="hybrid keeps searching after H=1 and accepts H=2 on the probability alone")
    def test_hybrid_without_cegis_finds_h_1(self, monkeypatch):
        """Without CEGIS, which may find H=1 at the root first, hybrid searches like AR."""
        monkeypatch.setattr(paynt.synthesizer.synthesizer_hybrid.StageControl, "only_ar", True)
        factory, task = load_sketch("tests/constraint-decided-by-one-design", sketch_name="hybrid.templ", props_name="hybrid.props")
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(factory.build(), task, "hybrid").run()
        assert str(result.assignment) == "H=1"
