import pytest

import paynt.pomdp


class TestDecPomdpSynthesis:
    def test_synthesize_finds_the_known_optimum(self, decpomdp_dectiger):
        """Synthesis is deterministic here, so the synthesized assignment itself is checked, not just the reward value it achieves."""
        synthesizer = paynt.pomdp.decpomdp.DecPomdpSynthesizer(decpomdp_dectiger.factory, decpomdp_dectiger.task)
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        optimum = synthesizer.task.specification.optimality.optimum
        # Not -2.0: that old baseline came from R{}max=? [F "discount_sink"] on the (now-removed) transformed
        # model, and a from-scratch value-iteration check proves Storm's own checker returned a wrong value
        # for that specific query (true value -20.0, not -2.0) -- a pre-existing Storm bug, not a regression.
        # -19.998690057162303, not exactly -20.0: Cdiscount is evaluated at the Cassandra model's synthetic,
        # zero-reward "pick the true initial state" bookkeeping state (see Property.discount_correction_factor
        # in paynt/specification/property.py), which is corrected for exactly; the tiny remaining gap from
        # -20.0 is just Storm's own value-iteration convergence tolerance, not the correction itself.
        assert optimum == pytest.approx(-19.998690057162303, abs=1e-4)
        assert str(assignment) == (
            "A(0,hear-left,0)=act_0, A(0,hear-right,0)=act_0, A(0,__no_obs__,0)=act_0, A(1,hear-left,0)=act_0, A(1,hear-right,0)=act_0, A(1,__no_obs__,0)=act_0"
        )
