"""Phase 1 differential tests for ColoringGeneral (payntbind.synthesis.ColoringGeneral): lift several tracked "generic"/"family" fixtures' plain Coloring via
fromColoring and check it agrees with the original Coloring, then check the generic engines (CEGIS/OneByOne/SMPMC) reach the same feasibility answer on the
lifted colored MDP.

Lifting a plain Coloring is not meant to compete with Coloring on performance -- these tests check correctness only, on fixtures that don't themselves need
generality.
"""

from __future__ import annotations

import random

import pytest

import paynt.mdp_family
import paynt.mdp_family.pomdp._utils
import paynt.mdp_family.task
import paynt.synthesizer.smpmc
import paynt.synthesizer.smpmc._augment
import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_ar
import paynt.utils.scoring

from helpers.helper import general_colored_mdp, load_colored_mdp

FIXTURES = ["tests/smpmc-tiny", "tests/mdp-family-avoid-8-2-easy"]


def random_subfamilies(parameter_space, count, seed):
    """A handful of random subfamilies of parameter_space (each parameter independently narrowed to a random nonempty subset of its own options, or left alone),
    reproducibly."""
    rng = random.Random(seed)
    subfamilies = []
    for _ in range(count):
        subfamily = parameter_space.copy()
        for parameter in range(subfamily.num_parameters):
            options = parameter_space.parameter_options(parameter)
            if len(options) > 1 and rng.random() < 0.7:
                k = rng.randint(1, len(options))
                subfamily.parameter_set_options(parameter, sorted(rng.sample(options, k)))
        subfamilies.append(subfamily)
    return subfamilies


@pytest.mark.parametrize("project", FIXTURES)
class TestFromColoringMatchesPlainColoring:
    def test_select_compatible_choices_on_random_subfamilies(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        general_mdp = general_colored_mdp(colored_mdp)
        for subfamily in random_subfamilies(colored_mdp.parameter_space, count=20, seed=0):
            plain = colored_mdp.coloring.selectCompatibleChoices(subfamily.native)
            general = general_mdp.coloring.selectCompatibleChoices(subfamily.native)
            assert list(plain) == list(general)

    def test_select_compatible_choices_with_parent_selection_as_base_choices(self, project):
        """A child subfamily's selection, restricted to its parent's own (broader) selection as base_choices, must equal the unrestricted selection -- for both
        colorings, and identically to each other."""
        colored_mdp, _task = load_colored_mdp(project)
        general_mdp = general_colored_mdp(colored_mdp)
        parameter_space = colored_mdp.parameter_space
        rng = random.Random(1)
        for parent in random_subfamilies(parameter_space, count=5, seed=2):
            plain_parent_choices = colored_mdp.coloring.selectCompatibleChoices(parent.native)
            general_parent_choices = general_mdp.coloring.selectCompatibleChoices(parent.native)
            assert list(plain_parent_choices) == list(general_parent_choices)

            child = parent.copy()
            for parameter in range(child.num_parameters):
                options = parent.parameter_options(parameter)
                if len(options) > 1 and rng.random() < 0.7:
                    k = rng.randint(1, len(options))
                    child.parameter_set_options(parameter, sorted(rng.sample(options, k)))

            plain_unrestricted = colored_mdp.coloring.selectCompatibleChoices(child.native)
            plain_restricted = colored_mdp.coloring.selectCompatibleChoices(child.native, plain_parent_choices)
            general_unrestricted = general_mdp.coloring.selectCompatibleChoices(child.native)
            general_restricted = general_mdp.coloring.selectCompatibleChoices(child.native, general_parent_choices)
            assert list(plain_unrestricted) == list(plain_restricted)
            assert list(general_unrestricted) == list(general_restricted)
            assert list(plain_restricted) == list(general_restricted)

    def test_get_state_to_holes_matches(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        general_mdp = general_colored_mdp(colored_mdp)
        plain = [list(x) for x in colored_mdp.coloring.getStateToHoles()]
        general = [list(x) for x in general_mdp.coloring.getStateToHoles()]
        assert plain == general

    def test_build_assignment_gives_an_identical_model(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        general_mdp = general_colored_mdp(colored_mdp)
        if colored_mdp.feature_kind in ("family", "pomdp_family"):
            pytest.skip("build_assignment keeps genuine nondeterminism for families (fixing the environment doesn't fix the policy) -- not comparable this way")
        parameter_space = colored_mdp.parameter_space
        combinations = list(parameter_space.all_combinations())
        rng = random.Random(3)
        sample_size = min(5, len(combinations))
        for combination in rng.sample(combinations, sample_size):
            assignment = parameter_space.construct_assignment(combination)
            plain_model = colored_mdp.build_assignment(assignment)
            general_model = general_mdp.build_assignment(assignment)
            assert str(plain_model.model.transition_matrix) == str(general_model.model.transition_matrix)


@pytest.mark.parametrize("project", FIXTURES)
@pytest.mark.parametrize("method", ["onebyone", "cegis", "smpmc"])
class TestEnginesAgreeOnLiftedColoredMdp:
    def test_feasibility_matches_ar_on_the_original_coloring(self, project, method):
        """CEGIS/OneByOne/SMPMC on the ColoringGeneral-lifted colored MDP must agree with AR on the original plain Coloring on the same fixture -- feasibility
        only (matching TestSmpmcOnFamilyModel's own reasoning: --method ar on a "family" fixture routes to PolicyTreeSynthesizer, a different problem, so AR is
        only the right oracle for "generic" fixtures; for "family" every listed engine, including onebyone here, answers the plain existential question, so they
        may be compared directly)."""
        colored_mdp, task = load_colored_mdp(project)
        general_mdp = general_colored_mdp(colored_mdp)
        general_result = paynt.synthesizer.synthesizer.Synthesizer.for_method(general_mdp, task, method).run()

        oracle_colored_mdp, oracle_task = load_colored_mdp(project)
        oracle_method = "onebyone" if oracle_colored_mdp.feature_kind == "family" else "ar"
        oracle_result = paynt.synthesizer.synthesizer.Synthesizer.for_method(oracle_colored_mdp, oracle_task, oracle_method).run()

        assert general_result.success == oracle_result.success


@pytest.mark.parametrize("project", FIXTURES)
class TestHasGeneralColoring:
    def test_false_for_a_standard_coloring_and_true_once_lifted(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        assert colored_mdp.has_general_coloring is False
        assert general_colored_mdp(colored_mdp).has_general_coloring is True


class TestGenericEnginesRejectAGeneralColoring:
    """AR and Hybrid read (parameter, option) pairs back from the coloring (ColoredMdp.scheduler_selection, split scoring), which a ColoringGeneral does not
    have: they are rejected up front with a clear error, and scheduler_selection itself is a backstop for anything constructed directly."""

    @pytest.mark.parametrize("method", ["ar", "hybrid"])
    def test_for_method_rejects_ar_based_methods(self, method):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        with pytest.raises(NotImplementedError, match=f"cannot use a general coloring for method '{method}'.*use --method onebyone, cegis or smpmc"):
            paynt.synthesizer.synthesizer.Synthesizer.for_method(general_colored_mdp(colored_mdp), task, method)

    @pytest.mark.parametrize("method", ["onebyone", "cegis", "smpmc"])
    def test_for_method_still_builds_the_engines_that_support_it(self, method):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(general_colored_mdp(colored_mdp), task, method)
        assert synthesizer.colored_mdp.has_general_coloring

    @pytest.mark.parametrize("method", ["ar", "hybrid"])
    def test_a_standard_coloring_is_unaffected(self, method):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        synthesizer = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method)
        assert not synthesizer.colored_mdp.has_general_coloring

    def test_an_invalid_method_name_is_still_reported_as_such(self):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        with pytest.raises(ValueError, match="invalid method name"):
            paynt.synthesizer.synthesizer.Synthesizer.for_method(general_colored_mdp(colored_mdp), task, "nonsense")

    def test_scheduler_selection_is_a_backstop(self):
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        with pytest.raises(NotImplementedError, match="scheduler_selection"):
            general_colored_mdp(colored_mdp).scheduler_selection(None, None)

    def test_running_ar_directly_fails_with_the_clear_error_not_an_attribute_error(self):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(general_colored_mdp(colored_mdp), task)
        with pytest.raises(NotImplementedError, match="scheduler_selection"):
            synthesizer.run()


class TestPairListConsumersRejectAGeneralColoring:
    """Everything that reads (parameter, option) pairs back from the coloring (getChoiceToAssignment, collectHoleOptions, ...) says so, clearly, for a
    ColoringGeneral -- instead of dying on an AttributeError or TypeError from deep inside payntbind."""

    def test_prob_goal_constraints(self):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        task.constraint_name = "prob1"
        synthesizer = paynt.synthesizer.smpmc.SynthesizerSMPMC(general_colored_mdp(colored_mdp), task)
        with pytest.raises(NotImplementedError, match="prob0/prob1 constraints"):
            synthesizer.run()

    def test_prob_goal_constraints_are_unaffected_for_a_standard_coloring(self):
        colored_mdp, task = load_colored_mdp("tests/smpmc-tiny")
        task.constraint_name = "prob1"
        assert paynt.synthesizer.smpmc.SynthesizerSMPMC(colored_mdp, task).run().success

    def test_adding_policy_parameters_for_robust_synthesis(self):
        colored_mdp, _task = load_colored_mdp("tests/mdp-family-avoid-8-2-easy")
        with pytest.raises(NotImplementedError, match="robust synthesis"):
            paynt.synthesizer.smpmc._augment.add_policy_parameters(general_colored_mdp(colored_mdp))

    def test_robust_smpmc_on_a_family(self):
        colored_mdp, task = load_colored_mdp("tests/mdp-family-avoid-8-2-easy")
        task.constraint_name = "exists_forall"
        with pytest.raises(NotImplementedError, match="robust synthesis"):
            paynt.synthesizer.smpmc.SynthesizerSMPMC(general_colored_mdp(colored_mdp), task)

    def test_policy_tree(self):
        colored_mdp, task = load_colored_mdp("tests/mdp-family-avoid-8-2-easy")
        with pytest.raises(NotImplementedError, match="policy tree"):
            paynt.mdp_family.PolicyTreeSynthesizer(general_colored_mdp(colored_mdp), task)

    def test_scheduler_memory_unfolding(self):
        colored_mdp, _task = load_colored_mdp("tests/mdp-family-avoid-8-2-easy")
        general = general_colored_mdp(colored_mdp)
        with pytest.raises(NotImplementedError, match="unfolding scheduler memory"):
            paynt.mdp_family.MdpFamilyColoredMdpFactory(
                colored_mdp.underlying_mdp, colored_mdp.parameter_space, general.coloring, paynt.mdp_family.task.MdpFamilyTask(memory_size=2)
            )

    def test_fsc_product(self):
        colored_mdp, _task = load_colored_mdp("tests/pomdp-family-avoid-smaller")
        with pytest.raises(NotImplementedError, match="FSC product"):
            paynt.mdp_family.pomdp._utils.build_dtmc_sketch(general_colored_mdp(colored_mdp), None)

    def test_ar_split_scoring(self):
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        with pytest.raises(NotImplementedError, match="split scoring"):
            paynt.utils.scoring.estimate_scheduler_difference(general_colored_mdp(colored_mdp), None, [], {}, [], [])

    def test_the_error_says_what_to_do_instead_when_there_is_something(self):
        colored_mdp, _task = load_colored_mdp("tests/smpmc-tiny")
        with pytest.raises(NotImplementedError, match="use --method onebyone, cegis or smpmc"):
            general_colored_mdp(colored_mdp).scheduler_selection(None, None)
