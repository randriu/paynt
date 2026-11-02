import os
import subprocess
import sys
import textwrap

from tests.helpers.models import get_sketch_paths

_RUN_SCRIPT = textwrap.dedent("""
    import sys

    import paynt.parser.sketch
    import paynt.synthesizer.synthesizer

    sketch_path, props_path = sys.argv[1], sys.argv[2]
    colored_mdp_factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp_factory.build(), task, "smpmc")
    result = synthesizer.run()
    assert result.success, "expected a satisfying assignment to be found"
    """)


class TestSynthesizerSmpmcCli:
    """SMPMC through the library entry point and the command line, each run in a process of its own, so that a crash or an abort at interpreter exit fails the
    test and not the whole test run."""

    def test_does_not_crash_on_threshold_property(self):
        sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
        result = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_does_not_crash_on_optimality_property(self):
        sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny", props_name="optimality.props")
        result = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_does_not_crash_on_family_model(self):
        sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
        result = subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr

    def test_robust_synthesis_exits_cleanly(self):
        """MBQI's fresh() propagators must not hold terms through Z3 sub-contexts, which Z3 frees once their round ends: that corrupts the heap at interpreter
        exit (see SmpmcPropagator.owner_ctx).

        Only a separate process sees it, and not on every run, as it depends on the heap layout; test_robust.py checks the invariant itself.
        """
        project = os.path.dirname(get_sketch_paths("tests/mdp-family-rocks-4-2")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--constraint", "exists_forall", "--smpmc-verify-robust"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "robustness verified by AR: True" in result.stdout

    def test_robust_optimization_reports_a_verified_worst_case(self):
        project = os.path.dirname(get_sketch_paths("tests/mdp-family-correlated")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--constraint", "exists_forall", "--smpmc-verify-robust"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "robust assignment, worst case 0.45" in result.stdout
        assert "robustness verified by AR: True" in result.stdout

    def test_decision_tree_synthesis_with_the_smpmc_engine(self):
        """--method smpmc on a decision-tree sketch runs SMPMC as the inner engine of the tree search, on the tree's general coloring."""
        project = os.path.dirname(get_sketch_paths("tests/dt-orchard")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--tree-depth", "0"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert "method: SMPMC" in result.stdout
        assert "the synthesized tree has value 0.4845045" in result.stdout

    def test_dtnest_has_no_smpmc_engine(self):
        project = os.path.dirname(get_sketch_paths("tests/dt-orchard")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--dtnest"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "--dtnest does not support method 'smpmc'" in result.stderr

    def test_costs_constraint_from_the_command_line(self):
        """--constraint costs reads sketch.costs of the project; 25 is exactly the cost of the cheapest satisfying assignment (see test_costs.py)."""
        project = os.path.dirname(get_sketch_paths("tests/smpmc-tiny")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--constraint", "costs", "--costs-threshold", "25"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert "x1=1, x2=4, x3=8" in result.stdout

    def test_costs_constraint_below_the_cheapest_assignment_is_infeasible(self):
        project = os.path.dirname(get_sketch_paths("tests/smpmc-tiny")[0])
        result = subprocess.run(
            [sys.executable, "-m", "paynt", project, "--method", "smpmc", "--constraint", "costs", "--costs-threshold", "24"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert "feasible: no" in result.stdout
