"""
Colored MDP (Definition 2 of arXiv:2511.08078): a colored MDP C = (M, V, kappa) is an MDP M, a constrained
parameter space V, and a coloring kappa: S x Act subseteq V such that for every theta in V and state s in S
it holds that there's exactly one action a such that theta in kappa(s, a).

Here the coloring may also be incomplete: a state may keep several choices under a full assignment theta (say, choices that no parameter colors), so that
C[theta] is an MDP and not a Markov chain. The specification is then to hold for the best resolution of the choices theta leaves, in the direction of each
property -- as if every such state had a policy parameter of its own that model checking, rather than the search, sets. Definition 2 is the case where
nothing is left to resolve. Incomplete coloring is supported for a single property; and not by robust synthesis, whose policy has to be explicit (see
ColoredMdp.build_assignment).

This is the only concrete ColoredMdp class. `feature_kind` (set at construction, e.g. "dt"/"pomdp"/"posmg"/
"family"/"pomdp_family"/"decpomdp") is what dispatch code and the 3 feature-dependent branches below key off,
instead of an isinstance check. Feature-specific state lives on
a small plain dataclass attached as `self.feature_info` (e.g. paynt.pomdp._utils.PomdpInfo) -- typed Any
here deliberately, so this module never needs to import any feature package.
"""

from __future__ import annotations

from typing import Any

import stormpy

import paynt.task
import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.model.model
import paynt.synthesizer.search_node
import paynt.utils.coloring
import paynt.utils.conditional
import paynt.utils.error_handling

import logging

logger = logging.getLogger(__name__)


class ColoredMdp:
    # label associated with un-labelled choices, shared by every coloring-construction implementation
    EMPTY_LABEL = "__no_label__"

    def __init__(
        self,
        underlying_mdp: Any,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        coloring: Any,
        use_exact: bool = False,
        feature_kind: str = "generic",
    ):
        # M: the underlying (uncolored) MDP, a stormpy sparse model
        self.underlying_mdp = underlying_mdp
        # V: the constrained parameter space
        self.parameter_space = parameter_space
        # kappa: raw payntbind coloring object (Coloring, ColoringSmt or ColoringGeneral); no Python wrapper exists for this
        self.coloring = coloring
        self.use_exact = use_exact
        # discriminator used by dispatch code to pick the right feature package without an isinstance check
        self.feature_kind = feature_kind
        # feature-specific companion data (e.g. paynt.pomdp._utils.PomdpInfo), attached by the factory;
        # None for "generic" and for features with no extra state (e.g. "decpomdp")
        self.feature_info: Any = None

        # internal plumbing needed by build()/scheduler_selection() below, not part of the public contract
        self.subsystem_builder_options = paynt.model.model.SubmodelBuilder.default_builder_options()
        self.choice_destinations = paynt.model.model.ModelIndex.compute_choice_destinations(underlying_mdp)
        # the parameters in the colors of the choices of each state, see relevant_parameters
        self._state_to_parameters: Any = None
        # see states_without_a_choice
        self._choice_coverage: paynt.utils.coloring.ChoiceCoverage | None = None
        # see choice_colors
        self._choice_to_assignment: list[list[tuple[int, int]]] | None = None

    @property
    def has_general_coloring(self) -> bool:
        """Whether kappa is a payntbind ColoringGeneral (an arbitrary formula per choice, see paynt.utils.coloring_builder) rather than a coloring given by
        explicit (parameter, option) pairs.

        Operations that read those pairs back from the coloring are unavailable for a general coloring, e.g. scheduler_selection (they fail through
        paynt.utils.error_handling.require_pair_list_coloring).
        """
        return paynt.utils.coloring.is_general_coloring(self.coloring)

    def build(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, parent_selected_choices: Any = None
    ) -> tuple[paynt.model.model.SubMdp, Any]:
        """Compute the induced sub-MDP C[eta] for the given parameter (sub)space.

        When some designs of eta reach a state that eta leaves without a choice, the sub-MDP keeps only the choices that keep away from such states: it then
        holds every design of eta that is not skipped, though not every design it holds is one of those (a design may reach a state that keeps choices of other
        designs alone).

        :param parent_selected_choices: optional reuse hint: the parent search node's own selected_choices (its compatible-choices bitmask). Since
            parameter_space is always a narrowing of the parent's (a child never widens what its parent already assumed), any choice compatible with
            parameter_space must already have been compatible with the parent -- so restricting the search to parent_selected_choices is sound and cannot miss a
            choice, only skip ones already known incompatible with an ancestor. None for the search root, which has no parent to reuse.
        :returns: (mdp, selected_choices)
        :raises paynt.utils.error_handling.DeadlockedDesigns: if every design of eta is skipped, or eta is a single design that is
        """
        if parent_selected_choices is None:
            choices = self.coloring.selectCompatibleChoices(parameter_space.native)
        else:
            choices = self.coloring.selectCompatibleChoices(parameter_space.native, parent_selected_choices)
        try:
            mdp = paynt.model.model.SubmodelBuilder.build_submdp(self.underlying_mdp, choices, self.subsystem_builder_options)
        except paynt.utils.error_handling.StateWithoutChoice:
            choices, reached = paynt.utils.error_handling.choices_avoiding_deadlocks(self, choices, parameter_space)
            mdp = paynt.model.model.SubmodelBuilder.build_submdp(self.underlying_mdp, choices, self.subsystem_builder_options)
            mdp.deciding_states = reached
        return mdp, choices

    def build_assignment(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> paynt.model.model.SubMdp:
        """Compute the induced model C[theta] for a full parameter assignment: a DTMC if theta leaves one choice in every state, else an MDP.

        An MDP results from an incomplete coloring, and always for "family"/"pomdp_family", where fixing the environment does not also fix the agent's policy.
        The remaining choices are then resolved by model checking, in the direction of the property (existentially). That is sound for a single property only,
        see Mdp.check_specification; robust synthesis rejects it, as its policy is a set of parameters chosen before the environment.

        :raises paynt.utils.error_handling.DeadlockedDesigns: if theta leaves a state it reaches without a choice
        """
        assert parameter_space.size == 1, "expecting parameter space of size 1"
        choices = self.coloring.selectCompatibleChoices(parameter_space.native)
        with paynt.utils.error_handling.raise_deadlocked_design(self, choices, parameter_space):
            model, state_map, choice_map = paynt.model.model.SubmodelBuilder.restrict(self.underlying_mdp, choices, self.subsystem_builder_options)
        if self.feature_kind in ("family", "pomdp_family") or model.nr_choices != model.nr_states:
            return paynt.model.model.SubMdp(model, state_map, choice_map)
        assert choices.get_number_of_set_bits() > 0
        dtmc = paynt.model.model.SubmodelBuilder.mdp_to_dtmc(model)
        return paynt.model.model.SubMdp(dtmc, state_map, choice_map)

    def selected_choices(
        self, assignment: paynt.parameter_space.parameter_space.ParameterSpace, specification: paynt.specification.property.Specification
    ) -> Any:
        """The choices of the underlying MDP that the solution assignment stands for: a mask with the choice taken in every state that it reaches.

        If theta leaves one choice per state, those are its choices. If it leaves several (an incomplete coloring, see build_assignment), it is the choices of a
        best policy for the single property of the specification -- in the direction that satisfies it, as when it was checked -- so the uncolored choices it
        takes are in the mask too. None if that property is step-bounded: its best policy depends on the step, and Storm gives none.
        """
        model = self.build_assignment(assignment)
        if model.is_deterministic:
            choices = stormpy.datastructures.BitVector(self.underlying_mdp.nr_choices, False)
            for choice in model.underlying_mdp_choice_map:
                choices.set(choice, True)
            return choices
        prop = specification.optimality if specification.optimality is not None else specification.constraints[0]
        if prop.is_step_bounded:
            logger.warning(f"no policy is reported for {assignment}: the best policy for {str(prop).strip()} depends on the step, and Storm gives none")
            return None
        scheduler = model.model_check_property(prop).result.get_scheduler()
        state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(self.underlying_mdp, self.choice_destinations, model, scheduler)
        return paynt.model.model.ModelIndex.state_to_choice_to_choices(self.underlying_mdp, state_to_choice)

    def scheduler_selection(self, mdp: Any, scheduler: Any, prop: paynt.specification.property.Property | None = None) -> list[list[int]]:
        """Get parameter options involved in the scheduler selection (the inverse of build(): choices -> V).

        For "posmg", unreachable choices are kept rather than discarded (unlike every other feature) since the induced model must still be verified as a game,
        not a plain MDP.

        Not available for a general coloring (see has_general_coloring), which has no per-parameter option lists to read back.

        :param prop: the property the scheduler is for; needed if it is a conditional property, whose scheduler has memory (see paynt.utils.conditional)
        """
        paynt.utils.error_handling.require_pair_list_coloring(
            self.coloring, "generic AR and Hybrid (scheduler_selection)", "use --method onebyone, cegis or smpmc"
        )
        if prop is not None and prop.is_conditional:
            choices = paynt.utils.conditional.scheduler_choice_mask(self.underlying_mdp, mdp, prop, scheduler)
            return self.coloring.collectHoleOptions(choices)
        assert scheduler.memoryless and scheduler.deterministic
        discard_unreachable_choices = self.feature_kind != "posmg"
        state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(
            self.underlying_mdp, self.choice_destinations, mdp, scheduler, discard_unreachable_choices=discard_unreachable_choices
        )
        choices = paynt.model.model.ModelIndex.state_to_choice_to_choices(self.underlying_mdp, state_to_choice)
        return self.coloring.collectHoleOptions(choices)

    def choice_colors(self) -> list[list[tuple[int, int]]]:
        """The (parameter, option) pairs of the color of each choice of the underlying MDP (for a pair-list coloring)."""
        if self._choice_to_assignment is None:
            self._choice_to_assignment = self.coloring.getChoiceToAssignment()
        return self._choice_to_assignment

    def relevant_parameters(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, states: Any) -> set[int]:
        """The parameters that decide which choices parameter_space selects in the given states of the underlying MDP: with any other parameter set differently,
        the same choices are selected there (Theorem 6 of arXiv:2511.08078)."""
        if self.has_general_coloring:
            return set(self.coloring.relevantParameters(parameter_space.native, list(states)))
        if self._state_to_parameters is None:
            self._state_to_parameters = self.coloring.getStateToHoles()
        relevant: set[int] = set()
        for state in states:
            relevant.update(self._state_to_parameters[state])
        return relevant

    def states_without_a_choice(self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, states: Any) -> list[int]:
        """The given states of the underlying MDP in which some design of parameter_space keeps no choice.

        A design that reaches one gets stuck, though the sub-MDP of several designs still holds the choices that the others take there.
        """
        if self._choice_coverage is None:
            self._choice_coverage = paynt.utils.coloring.ChoiceCoverage(self)
        return self._choice_coverage.states_without_a_choice(parameter_space, states)

    def find_deadlocked_design(
        self, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace
    ) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        """A design of parameter_space that leaves a state it reaches without a choice, or None if there is none.

        The parameter space is split only where its sub-MDP reaches a state that some of its designs leave without a choice.
        """
        spaces = [parameter_space]
        while spaces:
            space = spaces.pop()
            try:
                mdp, _choices = self.build(space)
            except paynt.utils.error_handling.DeadlockedDesigns:
                # every design of the space is skipped
                return space.pick_any()
            uncovered = self.states_without_a_choice(space, mdp.deciding_states)
            if not uncovered:
                continue
            # one design is never split: it keeps a choice in every state it reaches, or it is skipped above
            assert space.size > 1
            relevant = self.relevant_parameters(space, uncovered)
            splittable = [parameter for parameter in range(space.num_parameters) if space.parameter_num_options(parameter) > 1]
            splitter = next((parameter for parameter in splittable if parameter in relevant), splittable[0])
            spaces += space.split(splitter, space.suboptions_half(splitter))
        return None

    def build_from_choice_mask(self, choices: Any) -> paynt.model.model.SubMdp:
        """Restrict to a choice mask without needing a parameter space.

        Only "dt" uses this today (dtnest's subtree rebuilding), but the logic is generic -- it needs nothing feature-specific.
        """
        model, state_map, choice_map = paynt.model.model.SubmodelBuilder.restrict(self.underlying_mdp, choices, self.subsystem_builder_options)
        return paynt.model.model.SubMdp(model, state_map, choice_map)

    def are_choices_consistent(self, choices: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace) -> tuple[bool, list[list[int]]]:
        """Separate method for profiling purposes.

        Only "dt" uses this today, but the logic is generic -- it needs nothing feature-specific.
        """
        consistent, parameter_selection = self.coloring.areChoicesConsistent(choices, parameter_space.native)
        for parameter, options in enumerate(parameter_selection):
            assert len(options) == len(set(options)), str(parameter_selection)
            for option in options:
                assert option in parameter_space.parameter_options(parameter), (
                    f"option {option} for parameter {parameter} ({parameter_space.parameter_name(parameter)}) is not in the parameter space"
                )
        return consistent, parameter_selection

    def scheduler_is_consistent(
        self,
        mdp: Any,
        node: paynt.synthesizer.search_node.SearchNode,
        result: Any,
        specification: Any,
        prop: paynt.specification.property.Property | None = None,
    ) -> tuple[list[list[int]], bool]:
        """Get the parameter assignment induced by this scheduler and fill undefined parameters by some option from the parameter space of this mdp.

        :param node: the search node currently being verified -- provides parameter_space (the same one mdp was built from); node.scheduler_choices is
            additionally recorded on it for "dt", for the single-property scheduler-preservation shortcut
        :param specification: the specification currently being solved for -- unused except for "dt"
        :param prop: the property the scheduler is for, see scheduler_selection
        :return: parameter assignment
        :return: whether the scheduler is consistent (i.e. corresponds to exactly one assignment)
        """
        if mdp.is_deterministic:
            selection = [[node.parameter_space.parameter_options(parameter)[0]] for parameter in range(node.parameter_space.num_parameters)]
            return selection, True

        if self.feature_kind == "dt":
            # "dt" uses ColoringSmt's own consistency check instead of the generic scheduler_selection below
            scheduler = result.get_scheduler()
            assert scheduler.memoryless and scheduler.deterministic
            state_to_choice = paynt.model.model.ModelIndex.scheduler_to_state_to_choice(self.underlying_mdp, self.choice_destinations, mdp, scheduler)
            choices = paynt.model.model.ModelIndex.state_to_choice_to_choices(self.underlying_mdp, state_to_choice)
            if specification.is_single_property:
                node.scheduler_choices = choices  # type: ignore[attr-defined]
            consistent, parameter_selection = self.are_choices_consistent(choices, node.parameter_space)
            return parameter_selection, consistent

        # get qualitative scheduler selection, filter inconsistent assignments
        selection = self.scheduler_selection(mdp, result.get_scheduler(), prop)
        inconsistent_assignments = {parameter: options for parameter, options in enumerate(selection) if len(options) > 1}
        scheduler_is_consistent = len(inconsistent_assignments) == 0
        for parameter, options in enumerate(selection):
            if len(options) == 0:
                # if some parameter options are not involved in the selection, we can fix an arbitrary value
                selection[parameter] = [node.parameter_space.parameter_options(parameter)[0]]

        return selection, scheduler_is_consistent


class IdentityColoredMdpFactory:
    """Trivial "factory" for the plain generic ColoredMdp case (e.g. a DTMC with holes sketch): unlike every other feature, this ColoredMdp is built directly
    rather than through a real factory, and never needs re-unfolding at a different depth/memory size.

    Exists purely so Sketch.load_sketch/paynt.api.get_synthesizer can treat every feature uniformly as a (colored_mdp_factory, task) pair without special-casing
    this one.
    """

    feature_kind = "generic"

    def __init__(self, colored_mdp: ColoredMdp, task: paynt.task.SynthesisTask):
        self.colored_mdp = colored_mdp
        self.task = task

    def build(self) -> ColoredMdp:
        """Nothing to build -- the ColoredMdp already exists."""
        return self.colored_mdp
