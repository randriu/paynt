"""Phase 3: SynthesizerSMPMC as DtSynthesizer's inner engine (over the tree's ColoringGeneral), instead of the default SynthesizerARDt (over ColoringSmt) -- see
paynt.dt._utils and DtColoredMdpFactory.reset_tree.

Depth 0 is compared against SynthesizerARDt directly. Deeper trees are compared against optima pinned from SynthesizerARDt over ColoringSmt (see
tests/dt/test_dt_coloring_general.py's docstring on why running the AR engine over the general coloring is not a fair comparison there), and only became
feasible for SMPMC once its conflicts are minimized with ColoringGeneral.relevantParameters.
"""

from __future__ import annotations

import random

import pytest

import paynt.api
import paynt.model.model
import paynt.parser.sketch
import paynt.dt._utils
import paynt.dt.api
import paynt.dt.dtnest
import paynt.dt.result
import paynt.dt.synthesizer
import paynt.dt.synthesizer_ar_dt
import paynt.dt.task
import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc.checker
import paynt.utils.timer
from helpers.helper import get_sketch_paths

PROJECTS = ["tests/dt-orchard", "tests/dt-maze"]


@pytest.mark.parametrize("project", PROJECTS)
class TestSmpmcInnerEngineMatchesArDtAtDepthZero:
    def test_synthesize_tree_reaches_the_same_optimum(self, project):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        smpmc_synth = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        smpmc_synth.synthesize_tree(0)

        sketch_path, props_path = get_sketch_paths(project)
        factory_ar, task_ar = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_synth = paynt.dt.synthesizer.DtSynthesizer(factory_ar, task_ar, method="ar")
        ar_synth.synthesize_tree(0)

        assert smpmc_synth.best_tree is not None
        assert smpmc_synth.best_tree_value == pytest.approx(ar_synth.best_tree_value, abs=1e-6)

    def test_synthesize_tree_sequence_reaches_the_same_optimum(self, project):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        smpmc_synth = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        smpmc_synth.synthesize_tree_sequence(opt_result_value=0.0, overall_timeout=20, max_depth=1)

        sketch_path, props_path = get_sketch_paths(project)
        factory_ar, task_ar = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_synth = paynt.dt.synthesizer.DtSynthesizer(factory_ar, task_ar, method="ar")
        ar_synth.synthesize_tree_sequence(opt_result_value=0.0, overall_timeout=20, max_depth=1)

        assert smpmc_synth.best_tree_value == pytest.approx(ar_synth.best_tree_value, abs=1e-6)


class TestApiRouting:
    def test_get_synthesizer_routes_method_smpmc_to_dt_synthesizer_with_smpmc_inner_engine(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.api.get_synthesizer(factory, task, method="smpmc")
        assert isinstance(synthesizer, paynt.dt.synthesizer.DtSynthesizer)
        assert synthesizer.method == "smpmc"

    def test_dtnest_rejects_method_smpmc(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        with pytest.raises(ValueError, match="dtnest"):
            paynt.api.get_synthesizer(factory, task, method="smpmc", dtnest=True)

    def test_dtnest_with_the_default_method_routes_to_dtnest(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, task_kwargs={"tree_depth": 0})
        assert isinstance(paynt.api.get_synthesizer(factory, task, dtnest=True), paynt.dt.dtnest.DtNest)

    def test_mapping_a_scheduler_rejects_method_smpmc(self):
        """What --tree-map-scheduler sets: a scheduler file to map.

        SMPMC has nothing to search there.
        """
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        factory.build_task.scheduler_path = "scheduler.json"
        with pytest.raises(ValueError, match="mapping a scheduler to a tree does not support method 'smpmc'"):
            paynt.api.get_synthesizer(factory, task, method="smpmc")
        assert paynt.api.get_synthesizer(factory, task, method="ar").method == "ar"

    def test_a_generic_method_on_a_dt_sketch_is_rejected_with_a_clear_error(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        with pytest.raises(ValueError, match="decision-tree synthesis supports"):
            paynt.api.get_synthesizer(factory, task, method="onebyone")

    def test_default_method_ar_is_unaffected(self):
        """Regression guard: get_synthesizer's default routing for "dt" (no explicit --method) must still build a ColoringSmt-backed DtSynthesizer, exactly as
        before this feature existed."""
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.api.get_synthesizer(factory, task)
        assert isinstance(synthesizer, paynt.dt.synthesizer.DtSynthesizer)
        assert synthesizer.method == "ar"
        assert synthesizer.colored_mdp.coloring.__class__.__name__ == "ColoringSmt"


class TestSmpmcOnDeeperTrees:
    """Optima pinned from SynthesizerARDt over ColoringSmt; SMPMC proves optimality, so agreeing with that independent engine is the check.

    With the static per-state supports (every tree parameter, for every state) SMPMC's conflicts never shrink: dt-orchard depth 1 took minutes instead of
    seconds.
    """

    @pytest.mark.parametrize(
        ("project", "depth", "optimum"),
        [
            ("tests/dt-orchard", 1, 0.4882931556949054),
            ("tests/dt-maze", 1, 114.8076923076909),
            ("tests/dt-maze", 2, 20.948148711838865),
        ],
    )
    def test_reaches_the_optimum(self, project, depth, optimum):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        synthesizer.synthesize_tree(depth, timeout=120)
        assert synthesizer.best_tree is not None
        assert synthesizer.best_tree_value == pytest.approx(optimum, rel=1e-9)


class TestTreeSequenceOnDeeperTrees:
    """synthesize_tree_sequence tries every depth up to max_depth - 1, each seeded with the best tree of the depths before: the search runs on the subspace
    that hint describes first and on the whole space after, which depth 0 never does."""

    def test_the_depth_2_optimum_of_the_maze_is_reached_through_the_shallower_depths(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-maze")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        smpmc_synth = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        smpmc_synth.synthesize_tree_sequence(opt_result_value=0.0, overall_timeout=600, max_depth=3)
        assert smpmc_synth.best_tree is not None
        assert smpmc_synth.best_tree.get_depth() == 2
        assert smpmc_synth.best_tree_value == pytest.approx(20.948148711838865, rel=1e-9)

        factory_ar, task_ar = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        ar_synth = paynt.dt.synthesizer.DtSynthesizer(factory_ar, task_ar, method="ar")
        ar_synth.synthesize_tree_sequence(opt_result_value=0.0, overall_timeout=600, max_depth=3)
        assert smpmc_synth.best_tree_value == pytest.approx(ar_synth.best_tree_value, abs=1e-6)

    def test_a_tree_that_is_good_enough_ends_the_sequence_early(self):
        """opt_result_value is what the unrestricted optimum is; a tree within 0.1 % of it needs no deeper one."""
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        synthesizer = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        synthesizer.synthesize_tree_sequence(opt_result_value=0.4845, overall_timeout=300, max_depth=3)
        assert synthesizer.best_tree.get_depth() == 0
        assert synthesizer.best_tree_value == pytest.approx(0.48450450140758644, rel=1e-9)


class TestConstraintProperty:
    """A threshold instead of an optimum: a tree exists exactly when the threshold is within what its depth can reach.

    The orchard's optima, pinned above, are 0.4845... at depth 0 and 0.4882... at depth 1.
    """

    @staticmethod
    def synthesize(tmp_path, threshold, depth):
        props_path = tmp_path / "threshold.props"
        props_path.write_text(f"P>={threshold} [F goal]\n")
        sketch_path, _ = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, str(props_path))
        synthesizer = paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc")
        synthesizer.synthesize_tree(depth, timeout=600)
        return synthesizer, task

    @pytest.mark.parametrize(("threshold", "depth", "satisfiable"), [(0.48, 0, True), (0.488, 0, False), (0.49, 0, False), (0.488, 1, True)])
    def test_a_tree_is_found_exactly_when_the_threshold_is_reachable(self, tmp_path, threshold, depth, satisfiable):
        synthesizer, task = self.synthesize(tmp_path, threshold, depth)
        assert (synthesizer.best_tree is not None) is satisfiable
        # a constraint has no value to report, as for every other engine
        assert synthesizer.best_tree_value is None

    def test_the_tree_found_satisfies_the_constraint(self, tmp_path):
        synthesizer, task = self.synthesize(tmp_path, 0.488, 1)
        inner = paynt.dt._utils.make_inner_synthesizer("smpmc", synthesizer.colored_mdp, task)
        inner.synthesize(keep_optimum=True)
        assert inner.best_assignment is not None
        dtmc = synthesizer.colored_mdp.build_assignment(inner.best_assignment)
        result = dtmc.check_specification(task.specification)
        assert result.constraints_result.sat
        assert result.constraints_result.results[0].value >= 0.488


class TestDtSynthesizerRun:
    """The whole driver, as the command line runs it: the unrestricted optimum, the tree, its simplification and the result."""

    @pytest.fixture(autouse=True)
    def global_timer(self, monkeypatch):
        monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)
        paynt.utils.timer.GlobalTimer.start(None)

    @pytest.mark.parametrize("method", ["ar", "smpmc"])
    def test_the_result_carries_the_tree_and_its_value(self, method):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        # the depth is a task option: the command line defaults it to 0, a library call to the (much deeper) default of the task
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, task_kwargs={"tree_depth": 0})
        result = paynt.dt.synthesizer.DtSynthesizer(factory, task, method=method).run()
        assert isinstance(result, paynt.dt.result.DtResult)
        assert result.success
        assert result.value == pytest.approx(0.48450450140758644, rel=1e-9)
        assert result.tree is not None and result.tree.get_depth() == 0

    def test_the_search_is_named_after_its_inner_engine(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, task_kwargs={"tree_depth": 0})
        assert paynt.dt.synthesizer.DtSynthesizer(factory, task, method="ar").method_name == "AR (decision tree)"
        assert paynt.dt.synthesizer.DtSynthesizer(factory, task, method="smpmc").method_name == "smpmc (decision tree)"


class TestInnerEngineFactory:
    def test_every_supported_method_builds_its_engine_over_its_own_coloring(self):
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        assert paynt.dt._utils.DT_INNER_METHODS == ("ar", "smpmc")
        ar = paynt.dt._utils.make_inner_synthesizer("ar", factory.reset_tree(0), task)
        assert isinstance(ar, paynt.dt.synthesizer_ar_dt.SynthesizerARDt)
        assert type(ar.colored_mdp.coloring).__name__ == "ColoringSmt"
        smpmc = paynt.dt._utils.make_inner_synthesizer("smpmc", factory.reset_tree(0, general=True), task)
        assert isinstance(smpmc, paynt.synthesizer.smpmc.SynthesizerSMPMC)
        assert type(smpmc.colored_mdp.coloring).__name__ == "ColoringGeneral"

    def test_a_method_listed_but_not_built_is_a_bug_and_says_so(self, monkeypatch):
        """DT_INNER_METHODS and make_inner_synthesizer have to be extended together."""
        sketch_path, props_path = get_sketch_paths("tests/dt-orchard")
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        monkeypatch.setattr(paynt.dt._utils, "DT_INNER_METHODS", (*paynt.dt._utils.DT_INNER_METHODS, "cegis"))
        with pytest.raises(AssertionError, match="DT_INNER_METHODS has a method that make_inner_synthesizer does not build: 'cegis'"):
            paynt.dt._utils.make_inner_synthesizer("cegis", factory.reset_tree(0), task)


class TestConflictsOnTrees:
    """The soundness of ColoredMdpTheory's conflicts on a general coloring: whenever it learns a conflict smaller than the parameters fixed, every full
    assignment agreeing with the conflict must be refuted too (checked by model-checking the extension directly), or the learned clause would prune a real
    solution."""

    @pytest.mark.parametrize(
        ("project", "threshold", "polarity"),
        [
            # only trees needing fewer than 30 steps improve: most partial trees are refuted as `viable`
            ("tests/dt-maze", 30.0, True),
            # every tree improves: many partial trees are refuted as `not viable`, i.e. even their worst case improves
            ("tests/dt-maze", 1e6, False),
            ("tests/dt-orchard", 0.4884, True),
            ("tests/dt-orchard", 0.3, False),
        ],
    )
    def test_every_extension_of_a_smaller_conflict_is_refuted_too(self, project, threshold, polarity):
        sketch_path, props_path = get_sketch_paths(project)
        factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
        colored_mdp = factory.reset_tree(2, general=True)
        parameter_space = colored_mdp.parameter_space
        optimality = task.specification.optimality
        optimality.update_optimum(threshold)
        theory = paynt.synthesizer.smpmc.checker.ColoredMdpTheory(colored_mdp, optimality, None)

        rng = random.Random(1)
        smaller = 0
        for _ in range(100):
            fixed = {p: rng.choice(parameter_space.parameter_options(p)) for p in range(parameter_space.num_parameters) if rng.random() < 0.7}
            refutation = theory.check(fixed, polarity)
            if refutation is None or len(refutation.conflict_parameters) == len(fixed):
                continue
            smaller += 1
            conflict = set(refutation.conflict_parameters)
            assert conflict <= set(fixed)
            for _ in range(4):
                extension = parameter_space.copy()
                for p in range(parameter_space.num_parameters):
                    extension.parameter_set_options(p, [fixed[p] if p in conflict else rng.choice(parameter_space.parameter_options(p))])
                result = theory._model_check(colored_mdp.build_assignment(extension), alt=(polarity is False))
                assert result.sat is not polarity, f"extension of conflict {sorted(conflict)} is not refuted"
        assert smaller > 0, "no refutation had a conflict smaller than the fixed parameters: the check proves nothing about minimization"


def load(project):
    sketch_path, props_path = get_sketch_paths(project)
    return paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)


def map_optimal_scheduler(project, max_depth, method="ar"):
    """Map Storm's optimal scheduler of the project's MDP to a tree of at most max_depth, through the library entry point."""
    factory, task = load(project)
    scheduler = paynt.model.model.Mdp(factory.underlying_mdp).model_check_property(task.get_property()).result.scheduler
    build_task = paynt.dt.task.DtTask(tree_depth=max_depth)
    build_task.set_scheduler_to_map(scheduler)
    return paynt.dt.api.synthesize(factory, task, build_task, method=method)


class TestMapScheduler:
    """Mapping a scheduler to a tree is one satisfiability query per depth, answered by ColoringSmt; there is nothing for SMPMC to search, so it is refused.

    That the tree makes exactly the scheduler's choices is checked where the query is answered, see TestAreChoicesConsistentMatchesColoringSmt.
    """

    def test_the_optimal_scheduler_of_the_maze_is_mapped_to_a_tree_of_depth_3(self):
        result = map_optimal_scheduler("tests/dt-maze", 4)
        assert result.success
        assert result.tree.get_depth() == 3
        assert len(result.tree.collect_nonterminals()) == 5
        assert result.value == pytest.approx(6.889153192908649, rel=1e-6)

    @pytest.mark.parametrize(("project", "max_depth"), [("tests/dt-maze", 2), ("tests/dt-orchard", 2)])
    def test_no_tree_is_found_when_the_optimal_scheduler_needs_a_deeper_one(self, project, max_depth):
        assert not map_optimal_scheduler(project, max_depth).success

    def test_method_smpmc_is_refused(self):
        with pytest.raises(ValueError, match="mapping a scheduler to a tree does not support method 'smpmc'"):
            map_optimal_scheduler("tests/dt-maze", 4, "smpmc")

    def test_the_method_is_checked(self):
        factory, task = load("tests/dt-maze")
        with pytest.raises(ValueError, match="decision-tree synthesis supports"):
            paynt.dt.synthesizer.DtSynthesizer(factory, task, method="onebyone")


class TestLibraryEntryPointTakesTheMethod:
    def test_synthesizing_trees_through_paynt_dt_api_with_smpmc(self):
        factory, task = load("tests/dt-maze")
        smpmc = paynt.dt.api.synthesize(factory, task, paynt.dt.task.DtTask(tree_depth=1), use_solver="dtpaynt", method="smpmc")
        factory, task = load("tests/dt-maze")
        ar = paynt.dt.api.synthesize(factory, task, paynt.dt.task.DtTask(tree_depth=1), use_solver="dtpaynt")
        assert smpmc.success and ar.success
        assert smpmc.value == pytest.approx(ar.value, abs=1e-6)
