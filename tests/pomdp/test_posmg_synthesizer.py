import pytest

import paynt.pomdp.posmg


class TestPosmgSynthesis:
    def test_synthesize_improves_with_memory_unfolding(self, posmg_test_game):
        """Regression test for the re-unfolding path specifically: test-game's optimum improves from ~0.375 at memory size 1 to ~0.439453 once the optimizing
        player gets a second memory state, and the specification's tracked optimum reflects it. mec-test wouldn't catch a broken re-unfold here since its
        observations are all single-state and memory size never changes anything for it.

        Synthesis is deterministic, so the synthesized assignment at each memory size is checked too, not just the value it achieves.
        """
        synthesizer = paynt.pomdp.posmg.PosmgSynthesizer(posmg_test_game.factory, posmg_test_game.task)

        assignment_k1 = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        optimum_k1 = synthesizer.task.specification.optimality.optimum
        assert optimum_k1 == pytest.approx(0.375, abs=1e-4)
        assert str(assignment_k1) == "A(P0,O1,M0)=dbl, A(P0,O2,M0)=add, A(P1,S0,M0)=even"

        synthesizer.colored_mdp = synthesizer.colored_mdp_factory.set_imperfect_memory_size(2)
        assignment_k2 = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
        optimum_k2 = synthesizer.task.specification.optimality.optimum
        assert optimum_k2 == pytest.approx(0.439453, abs=1e-4)
        assert optimum_k2 > optimum_k1
        assert str(assignment_k2) == (
            "A(P0,O1,M0)=dbl, A(P0,O1,M1)=add, M(P0,O1,M0)=1, M(P0,O1,M1)=0, "
            "A(P0,O2,M0)=add, A(P0,O2,M1)=add, M(P0,O2,M0)=1, M(P0,O2,M1)=1, "
            "A(P1,S0,M0)=even, A(P1,S0,M1)=odd"
        )
