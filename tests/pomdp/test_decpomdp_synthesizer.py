import pytest

import paynt.pomdp


class TestDecPomdpSynthesis:
    def test_synthesize_finds_the_known_optimum(self, decpomdp_dectiger):
        synthesizer = paynt.pomdp.decpomdp.DecPomdpSynthesizer(decpomdp_dectiger.factory, decpomdp_dectiger.task)
        assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        optimum = synthesizer.task.specification.optimality.optimum
        # solver output, a regression value: -20 up to the tolerance of Storm's value iteration
        assert optimum == pytest.approx(-19.998690057162303, abs=1e-4)
        assert str(assignment) == (
            "A(0,hear-left,0)=act_0, A(0,hear-right,0)=act_0, A(0,__no_obs__,0)=act_0, A(1,hear-left,0)=act_0, A(1,hear-right,0)=act_0, A(1,__no_obs__,0)=act_0"
        )
