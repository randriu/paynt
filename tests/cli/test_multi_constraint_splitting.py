import subprocess
import sys
import textwrap

import pytest

from tests.helpers.models import get_sketch_paths

_RUN_SCRIPT = textwrap.dedent("""
    import sys

    import paynt.parser.sketch
    import paynt.synthesizer.synthesizer

    sketch_path, props_path, method = sys.argv[1], sys.argv[2], sys.argv[3]
    colored_mdp_factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp_factory.colored_mdp, task, method)
    synthesizer.run()
    """)


class TestMultiConstraintSplitting:
    """
    Regression coverage for the multi-constraint AR splitting gap (paper Section https://www.jair.org/index.php/jair/article/view/16593 3.3, "AR for Feasibility
    Synthesis with Multiple Constraints": models/tests/generic-multiconstraint-dpm
    easy.props has zero optimality objective and exactly two constraints -- about as pure a multi-constraint
    case as exists -- and used to crash every AR-based method within the first few seconds/iterations: `ar`
    directly (ValueError: max() iterable argument is empty, from split_parameter_space's scheduler_scores
    returning an empty dict), `cegis` via an independent bug in collect_conflict_requests (AttributeError:
    'NoneType' object has no attribute 'sat', from short_evaluation=True leaving later constraint indices
    None), and `hybrid` via a related indexing bug in its priority-node heuristic (results[0] instead of
    undecided_constraints[0]).

    This model is too large (19606 states, design space 1e7) to synthesize to completion in a unit test
    (hours for ar/cegis; hybrid gets much further but still doesn't converge within a minute), so this only
    asserts survival past the original crash point, not a pinned final answer. Run out-of-process (rather
    than under a bounded GlobalTimer) because SynthesizerCEGIS.synthesize_one and SynthesizerHybrid.
    synthesize_one -- unlike SynthesizerAR.synthesize_one -- never call resource_limit_reached() in their
    search loops, so neither respects GlobalTimer/a per-call timeout at all; a real subprocess killed from
    the outside is the only reliable way to bound them. Every one of the three original crashes happened
    within the first few seconds/iterations (confirmed manually), so "no exception raised in 15s" is a
    meaningful regression signal, not a coincidence of timing.
    """

    @pytest.mark.parametrize("method", ["ar", "cegis", "hybrid"])
    def test_does_not_crash_on_a_genuinely_multi_constraint_specification(self, method):
        sketch_path, props_path = get_sketch_paths("tests/generic-multiconstraint-dpm", props_name="easy.props")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _RUN_SCRIPT, sketch_path, props_path, method],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            # still searching after 15s with no exception -- the expected outcome here, since none of the
            # three methods converge on this model within a unit-test time budget
            return
        assert result.returncode == 0, result.stderr
