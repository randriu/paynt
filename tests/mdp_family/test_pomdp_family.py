import pytest

import paynt.colored_mdp
import paynt.mdp_family
import paynt.mdp_family.pomdp
import paynt.mdp_family.pomdp._utils
import paynt.pomdp.fsc
import paynt.synthesizer.synthesizer
import paynt.task


class TestPomdpFamilyColoredMdpFactory:
    def test_factory_does_not_build_automatically(self, pomdp_family_avoid_smaller):
        """Guardrail for the "factories don't eagerly build" redesign: a freshly-constructed factory holds no colored_mdp of its own -- build() must be called
        explicitly to get one."""
        assert not hasattr(pomdp_family_avoid_smaller.factory, "colored_mdp")

    def test_load_sketch_produces_a_pomdp_family_colored_mdp(self, pomdp_family_avoid_smaller):
        assert type(pomdp_family_avoid_smaller.colored_mdp) is paynt.colored_mdp.ColoredMdp
        assert pomdp_family_avoid_smaller.colored_mdp.feature_kind == "pomdp_family"
        assert isinstance(pomdp_family_avoid_smaller.colored_mdp.feature_info, paynt.mdp_family.pomdp._utils.PomdpFamilyInfo)

    def test_observation_structure_is_populated(self, pomdp_family_avoid_smaller):
        num_observations = paynt.mdp_family.pomdp._utils.num_observations(pomdp_family_avoid_smaller.colored_mdp.feature_info)
        assert num_observations > 0
        assert len(pomdp_family_avoid_smaller.colored_mdp.feature_info.observation_to_actions) == num_observations
        state_to_observation = paynt.mdp_family.pomdp._utils.state_to_observation(pomdp_family_avoid_smaller.colored_mdp.feature_info)
        assert len(state_to_observation) == pomdp_family_avoid_smaller.colored_mdp.underlying_mdp.nr_states

    def test_build_pomdp_produces_a_sub_pomdp_for_a_single_environment(self, pomdp_family_avoid_smaller):
        assignment = pomdp_family_avoid_smaller.colored_mdp.parameter_space.pick_any()
        sub_pomdp = paynt.mdp_family.pomdp._utils.build_pomdp(pomdp_family_avoid_smaller.colored_mdp, assignment)
        assert sub_pomdp.model.nr_states > 0
        assert len(sub_pomdp.underlying_mdp_state_map) == sub_pomdp.model.nr_states


def _trivial_fsc(pomdp_family_avoid_smaller, ambiguous_action):
    """A single-node FSC: observations with only one legal action take it (fill_trivial_actions); the two ambiguous, multi-action observations both take
    `ambiguous_action`.

    build_dtmc_sketch requires a stochastic-form FSC (its native FscUnfolder binding rejects the bare int/deterministic form even when
    FscFactored.is_deterministic is True), hence make_stochastic() below -- this was never exercised before since build_dtmc_sketch had no callers anywhere in
    the codebase prior to this test.
    """
    num_observations = paynt.mdp_family.pomdp._utils.num_observations(pomdp_family_avoid_smaller.colored_mdp.feature_info)
    observation_to_actions = pomdp_family_avoid_smaller.colored_mdp.feature_info.observation_to_actions
    fsc = paynt.pomdp.fsc.FscFactored(1, num_observations, is_deterministic=True)
    fsc.fill_trivial_actions(observation_to_actions)
    for obs, actions in enumerate(observation_to_actions):
        if len(actions) > 1:
            fsc.action_function[0][obs] = ambiguous_action
        fsc.update_function[0][obs] = 0
    fsc.check(observation_to_actions)
    fsc.make_stochastic()
    return fsc


def _task_for_dtmc_sketch(pomdp_family_avoid_smaller):
    """build_dtmc_sketch produces a bare (task-less) ColoredMdp -- since the same product can be reused across different tasks, the caller supplies its own.

    Here we just carry over the original task's specification/timeout/use_exact unchanged (a copy, so synthesize()'s specification.reset() can't affect the
    original task).
    """
    return paynt.task.SynthesisTask.from_specification(
        pomdp_family_avoid_smaller.task.specification.copy(),
        timeout=pomdp_family_avoid_smaller.task.timeout,
        use_exact=pomdp_family_avoid_smaller.factory.use_exact,
    )


class TestPomdpFamilyDtmcSketch:
    def test_build_dtmc_sketch_produces_a_plain_colored_mdp(self, pomdp_family_avoid_smaller):
        """build_dtmc_sketch's whole point is to hand the FSC-fixed family off to the *generic* AR/CEGIS
        machinery -- so its output must be a plain, unspecialized ColoredMdp, not another "pomdp_family" one."""
        fsc = _trivial_fsc(pomdp_family_avoid_smaller, ambiguous_action=7)
        dtmc_sketch = paynt.mdp_family.pomdp._utils.build_dtmc_sketch(pomdp_family_avoid_smaller.colored_mdp, fsc)
        assert type(dtmc_sketch) is paynt.colored_mdp.ColoredMdp
        assert dtmc_sketch.feature_kind == "generic"
        assert dtmc_sketch.parameter_space.size == pomdp_family_avoid_smaller.colored_mdp.parameter_space.size

    def test_synthesize_over_the_fsc_fixed_family_finds_the_best_environment(self, pomdp_family_avoid_smaller):
        """End-to-end regression test for the build_dtmc_sketch -> generic AR pipeline: fixing a memoryless FSC across all 64 environment variants turns the
        POMDP family into a plain family of DTMCs, and AR then searches over which environment (not which policy -- that's already fixed) best matches this
        FSC's objective.

        Synthesis is deterministic, so both the value and the winning environment are checked.
        """
        fsc = _trivial_fsc(pomdp_family_avoid_smaller, ambiguous_action=7)
        dtmc_sketch = paynt.mdp_family.pomdp._utils.build_dtmc_sketch(pomdp_family_avoid_smaller.colored_mdp, fsc)
        task = _task_for_dtmc_sketch(pomdp_family_avoid_smaller)
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(dtmc_sketch, task, "ar")
        assignment = synthesizer.synthesize(print_stats=False, keep_optimum=True)
        assert assignment is not None
        assert str(assignment) == "o1x_init=1, o2x_init=1, goright1_init=0, goright2_init=0, o1y=2, o2y=2"
        assert task.specification.optimality.optimum == pytest.approx(4.0, abs=1e-6)
