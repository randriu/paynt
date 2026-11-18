"""paynt.utils.error_handling: the designs that leave a state without a choice, the errors for what an incomplete coloring does not support, for a general
coloring where the (parameter, option) pairs are needed, and for a method that a feature does not run.

The functions that read a few attributes of the colored MDP, the model and the specification are exercised on stand-ins; the same errors are reached through the
real classes in tests/coloring/test_incomplete_coloring.py, tests/coloring/test_coloring_general.py and tests/dt/test_smpmc_inner_engine.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import paynt.utils.error_handling as error_handling

from tests.helpers.models import general_colored_mdp, load_colored_mdp

# A sparse MDP of four states as Storm lays it out, choices 0 | 1 2 | 3 | 4: state 0 is initial, its choice leads to state 1, whose choices lead to states 2
# and 3, and states 2 and 3 lead to 3.
CHOICE_GROUPS = [0, 1, 3, 4, 5]
CHOICE_DESTINATIONS = [[1], [2], [3], [3], [3]]


def colored_mdp(choice_destinations=None, initial_states=(0,)):
    underlying_mdp = SimpleNamespace(
        nondeterministic_choice_indices=CHOICE_GROUPS, initial_states=list(initial_states), nr_choices=5, has_state_valuations=lambda: False
    )
    return SimpleNamespace(underlying_mdp=underlying_mdp, choice_destinations=choice_destinations or CHOICE_DESTINATIONS)


class Space:
    def __init__(self, size=1):
        self.size = size

    def __str__(self):
        return "H=1" if self.size == 1 else "H:{1,2}"


class TestRaiseDeadlockedDesign:
    def test_a_block_that_succeeds_is_left_alone(self):
        with error_handling.raise_deadlocked_design(colored_mdp(), [True] * 5, Space()):
            built = "the model"
        assert built == "the model"

    def test_names_the_state_left_without_a_choice_and_the_way_to_it(self):
        # state 1 is reached through choice 0 and has both of its choices disabled
        with pytest.raises(error_handling.DeadlockedDesigns, match=r"^skipping design H=1: state 1 is reachable and keeps no choice$") as raised:
            with error_handling.raise_deadlocked_design(colored_mdp(), [True, False, False, True, True], Space()):
                raise error_handling.StateWithoutChoice()
        assert (raised.value.state, raised.value.states) == (1, [0, 1])

    def test_the_way_is_a_shortest_one(self):
        # state 3 is reached through state 1's second choice, and its choice is disabled
        with pytest.raises(error_handling.DeadlockedDesigns) as raised:
            with error_handling.raise_deadlocked_design(colored_mdp(), [True, False, True, True, False], Space()):
                raise error_handling.StateWithoutChoice()
        assert raised.value.states == [0, 1, 3]

    def test_hides_the_signal_of_the_builder(self):
        with pytest.raises(error_handling.DeadlockedDesigns) as raised:
            with error_handling.raise_deadlocked_design(colored_mdp(), [True, False, False, True, True], Space()):
                raise error_handling.StateWithoutChoice()
        assert raised.value.__cause__ is None
        assert raised.value.__suppress_context__

    def test_any_other_failure_is_reported_as_it_is(self):
        with pytest.raises(RuntimeError, match="out of memory"):
            with error_handling.raise_deadlocked_design(colored_mdp(), [True, False, False, True, True], Space()):
                raise RuntimeError("out of memory")


class TestChoicesAvoidingDeadlocks:
    def test_drops_the_choices_into_a_state_without_one(self):
        # state 2 keeps no choice: state 1 goes to state 3 instead
        choices, reached = error_handling.choices_avoiding_deadlocks(colored_mdp(), [True, True, True, False, True], Space(2))
        assert list(choices) == [0, 2, 4]
        # which of them are kept depends on the choices of every state reached, state 2 included
        assert reached == [0, 1, 2, 3]

    def test_every_design_is_skipped_when_the_initial_state_keeps_no_choice(self):
        # state 2 keeps no choice, and state 1 only the one into it
        with pytest.raises(error_handling.DeadlockedDesigns, match=r"^skipping the 2 designs of H:\{1,2\}: state 2 is reachable") as raised:
            error_handling.choices_avoiding_deadlocks(colored_mdp(), [True, True, False, False, True], Space(2))
        # every state reached decides it
        assert raised.value.states == [0, 1, 2]

    def test_a_single_design_is_skipped(self):
        with pytest.raises(error_handling.DeadlockedDesigns, match=r"^skipping design H=1: state 2 is reachable") as raised:
            error_handling.choices_avoiding_deadlocks(colored_mdp(), [True, True, True, False, True], Space())
        assert raised.value.states == [0, 1, 2]


class TestRequireSingleProperty:
    @staticmethod
    def call(is_deterministic, num_properties):
        error_handling.require_single_property_on_nondeterministic_model(
            SimpleNamespace(is_deterministic=is_deterministic), SimpleNamespace(num_properties=num_properties)
        )

    def test_accepts_several_properties_on_a_markov_chain(self):
        self.call(True, 3)

    def test_accepts_one_property_on_a_model_with_nondeterminism(self):
        self.call(False, 1)

    def test_rejects_several_properties_on_a_model_with_nondeterminism(self):
        with pytest.raises(
            NotImplementedError, match=r"cannot check 2 properties on a model that still has nondeterminism.*only a single property is supported"
        ):
            self.call(False, 2)


class TestRequireMarkovChainForRobustSearch:
    @staticmethod
    def call(is_deterministic, policy_parameters):
        error_handling.require_markov_chain_for_robust_search(SimpleNamespace(is_deterministic=is_deterministic), policy_parameters)

    def test_accepts_a_markov_chain(self):
        self.call(True, [0, 1])

    def test_accepts_a_model_with_nondeterminism_when_the_search_is_not_robust(self):
        self.call(False, None)

    def test_rejects_a_model_with_nondeterminism_in_a_robust_search(self):
        with pytest.raises(NotImplementedError, match="robust synthesis does not support an incomplete coloring"):
            self.call(False, [0, 1])

    def test_a_robust_search_without_parameters_to_choose_is_still_robust(self):
        with pytest.raises(NotImplementedError):
            self.call(False, [])


@pytest.mark.parametrize("project", ["tests/smpmc-tiny", "tests/mdp-family-avoid-8-2-easy"])
class TestRequirePairListColoring:
    def test_accepts_a_standard_coloring(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        error_handling.require_pair_list_coloring(colored_mdp.coloring, "anything")
        error_handling.require_pair_list_coloring(colored_mdp.coloring, "anything", "do something else")

    def test_rejects_a_general_coloring_naming_what_needed_the_pairs(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        with pytest.raises(NotImplementedError) as error:
            error_handling.require_pair_list_coloring(general_colored_mdp(colored_mdp).coloring, "the policy tree")
        assert str(error.value) == "cannot use a general coloring for the policy tree: it has no (parameter, option) pairs to read back"

    def test_appends_the_alternative_when_there_is_one(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        with pytest.raises(NotImplementedError) as error:
            error_handling.require_pair_list_coloring(general_colored_mdp(colored_mdp).coloring, "the policy tree", "use another method")
        assert str(error.value).endswith("no (parameter, option) pairs to read back; use another method")


class TestRequireSupportedMethod:
    def test_accepts_a_supported_method(self):
        error_handling.require_supported_method("smpmc", ("ar", "smpmc"), "decision-tree synthesis")

    def test_names_what_was_asked_the_methods_it_runs_and_the_one_it_was_given(self):
        with pytest.raises(ValueError) as error:
            error_handling.require_supported_method("onebyone", ("ar", "smpmc"), "decision-tree synthesis")
        assert str(error.value) == "decision-tree synthesis supports method 'ar' or 'smpmc', got 'onebyone'"

    def test_a_single_supported_method(self):
        with pytest.raises(ValueError, match="supports method 'ar', got 'cegis'"):
            error_handling.require_supported_method("cegis", ["ar"], "something")


class TestRequireMethodAr:
    def test_accepts_ar(self):
        error_handling.require_method_ar("ar", "--dtnest")

    @pytest.mark.parametrize("method", ["smpmc", "cegis", "hybrid", "onebyone"])
    def test_rejects_any_other_method_naming_what_needs_ar(self, method):
        with pytest.raises(ValueError) as error:
            error_handling.require_method_ar(method, "mapping a scheduler to a tree")
        assert str(error.value) == f"mapping a scheduler to a tree does not support method '{method}': it always uses AR over ColoringSmt, use method 'ar'"
