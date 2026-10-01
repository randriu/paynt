from __future__ import annotations

from typing import Any

import stormpy
import payntbind

import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.model.model_builder
import paynt.utils.error_handling

import itertools

import logging

logger = logging.getLogger(__name__)


class CombinationColoring:
    """Dictionary of colors associated with different parameter combinations.

    Note: color 0 is reserved for general parameter-free objects.
    """

    def __init__(self):
        self.coloring: dict[Any, int] = {}
        self.reverse_coloring: list[Any] = [None]

    @property
    def num_colors(self) -> int:
        return len(self.coloring)

    def get_or_make_color(self, parameter_assignment: Any) -> int:
        new_color = self.num_colors + 1
        color = self.coloring.get(parameter_assignment, new_color)
        if color == new_color:
            self.coloring[parameter_assignment] = color
            self.reverse_coloring.append(parameter_assignment)
        return color


class JaniUnfolder:
    """Unfolder of parameter combinations into JANI program."""

    def __init__(
        self,
        prism: Any,
        parameter_expressions: list[list[Any]],
        specification: paynt.specification.property.Specification,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        use_exact: bool = False,
    ):

        logger.debug("constructing JANI program...")

        # pack properties and translate Prism to Jani
        properties_old = specification.all_properties()
        stormpy_properties = [p.property for p in properties_old]
        jani, properties = prism.to_jani(stormpy_properties)
        paynt.utils.error_handling.require_no_holes_in_state_values(
            jani, [parameter_space.parameter_name(index) for index in range(parameter_space.num_parameters)]
        )

        # upon translation, some properties may change their atoms, so we need to re-wrap all properties
        properties_unpacked = []
        for index, prop_old in enumerate(properties_old):
            prop_new = properties[index]
            if type(prop_old) is paynt.specification.property.Property:
                p = paynt.specification.property.Property(prop_new, use_exact)
            else:
                assert isinstance(prop_old, paynt.specification.property.OptimalityProperty)
                epsilon = prop_old.epsilon
                p = paynt.specification.property.OptimalityProperty(prop_new, epsilon, use_exact)
            properties_unpacked.append(p)
        self.specification = paynt.specification.property.Specification(properties_unpacked)
        self.jani_unfolded, edge_to_parameter_options = JaniUnfolder.unfold_jani(jani, parameter_space, parameter_expressions)

        logger.debug("constructing the underlying model...")
        underlying_mdp = paynt.model.model_builder.ModelBuilder.from_jani(self.jani_unfolded, self.specification, use_exact=use_exact)

        # associate each action of the underlying MDP with parameter options
        # reconstruct choice labels from choice origins
        logger.debug("associating choices of the underlying model with parameter assignments...")
        if use_exact:
            choice_is_valid, choice_to_parameter_options = payntbind.synthesis.janiMapChoicesToHoleAssignmentsExact(
                underlying_mdp, parameter_space.native, edge_to_parameter_options
            )
        else:
            choice_is_valid, choice_to_parameter_options = payntbind.synthesis.janiMapChoicesToHoleAssignments(
                underlying_mdp, parameter_space.native, edge_to_parameter_options
            )

        # handle conflicting colors
        num_choices_all = underlying_mdp.nr_choices
        num_choices_valid = choice_is_valid.number_of_set_bits()
        if num_choices_valid < num_choices_all:
            logger.debug(f"keeping {num_choices_valid}/{num_choices_all} choices with non-conflicting parameter assignments...")
            keep_unreachable_states = False
            subsystem_builder_options = stormpy.SubsystemBuilderOptions()
            subsystem_builder_options.build_action_mapping = True
            all_states = stormpy.storage.BitVector(underlying_mdp.nr_states, True)
            submodel_construction = stormpy.construct_submodel(underlying_mdp, all_states, choice_is_valid, keep_unreachable_states, subsystem_builder_options)
            underlying_mdp = submodel_construction.model
            choice_map = list(submodel_construction.new_to_old_action_mapping)
            choice_to_parameter_options = [choice_to_parameter_options[choice_map[choice]] for choice in range(underlying_mdp.nr_choices)]

        self.underlying_mdp = underlying_mdp
        self.choice_to_parameter_options = choice_to_parameter_options
        return

    @staticmethod
    def unfold_jani(
        jani: Any, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace, parameter_expressions: list[list[Any]]
    ) -> tuple[Any, dict[int, list[tuple[int, int]]]]:
        # ensure that jani.constants are in the same order as our parameters
        open_constants = [c for c in jani.constants if not c.defined]
        parameter_variables = [c.expression_variable for c in open_constants]
        assert len(open_constants) == parameter_space.num_parameters
        for parameter in range(parameter_space.num_parameters):
            assert parameter_space.parameter_name(parameter) == open_constants[parameter].name

        combination_coloring = CombinationColoring()
        jani_program = stormpy.JaniModel(jani)
        new_automata = {}
        for aut_index, automaton in enumerate(jani_program.automata):
            if not JaniUnfolder.automaton_has_parameters(automaton, parameter_variables):
                continue
            new_aut = JaniUnfolder.construct_automaton(automaton, parameter_variables, parameter_expressions, combination_coloring)
            new_automata[aut_index] = new_aut
        for aut_index, aut in new_automata.items():
            jani_program.replace_automaton(aut_index, aut)
        for parameter in range(parameter_space.num_parameters):
            jani_program.remove_constant(parameter_space.parameter_name(parameter))

        jani_program.set_model_type(stormpy.JaniModelType.MDP)
        jani_program.finalize()
        jani_program.check_valid()

        # collect label and color of each edge
        edge_to_parameter_options = {}
        for aut_index, automaton in enumerate(jani_program.automata):
            for edge_index, edge in enumerate(automaton.edges):
                if edge.color == 0:
                    continue
                global_index = jani_program.encode_automaton_and_edge_index(aut_index, edge_index)
                options = combination_coloring.reverse_coloring[edge.color]
                options = [(parameter_index, option) for parameter_index, option in enumerate(options) if option is not None]
                edge_to_parameter_options[global_index] = options

        return jani_program, edge_to_parameter_options

    @staticmethod
    def edge_parameters(edge: Any, parameter_variables: list[Any]) -> list[int]:
        variables = set()
        variables |= edge.guard.get_variables()
        for assignment in edge.template_edge.assignments:
            variables |= assignment.expression.get_variables()
        for dest in edge.destinations:
            variables |= dest.probability.get_variables()
            for assignment in dest.assignments:
                variables |= assignment.expression.get_variables()
        for dest in edge.template_edge.destinations:
            for assignment in dest.assignments:
                variables |= assignment.expression.get_variables()
        return [parameter for parameter, variable in enumerate(parameter_variables) if variable in variables]

    @staticmethod
    def automaton_has_parameters(automaton: Any, parameter_variables: list[Any]) -> bool:
        for edge in automaton.edges:
            if len(JaniUnfolder.edge_parameters(edge, parameter_variables)) > 0:
                return True
        return False

    @staticmethod
    def construct_automaton(
        automaton: Any, parameter_variables: list[Any], parameter_expressions: list[list[Any]], combination_coloring: CombinationColoring
    ) -> Any:
        new_aut = stormpy.storage.JaniAutomaton(automaton.name, automaton.location_variable)
        [new_aut.add_location(loc) for loc in automaton.locations]
        [new_aut.add_initial_location(idx) for idx in automaton.initial_location_indices]
        [new_aut.variables.add_variable(var) for var in automaton.variables]
        for edge in automaton.edges:
            new_edges = JaniUnfolder.construct_edges(edge, parameter_variables, parameter_expressions, combination_coloring)
            for new_edge in new_edges:
                new_aut.add_edge(new_edge)
        return new_aut

    @staticmethod
    def construct_edges(
        edge: Any, parameter_variables: list[Any], parameter_expressions: list[list[Any]], combination_coloring: CombinationColoring
    ) -> list[Any]:
        edge_parameters = JaniUnfolder.edge_parameters(edge, parameter_variables)
        if len(edge_parameters) == 0:
            return [JaniUnfolder.construct_edge(edge)]

        combinations = [
            (list(range(len(expressions))) if parameter in edge_parameters else [None]) for parameter, expressions in enumerate(parameter_expressions)
        ]
        new_edges = []
        for combination in itertools.product(*combinations):
            substitution = {
                # combination[parameter] is guarded non-None by the if-clause below, but mypy doesn't
                # narrow a comprehension's key/value expression from its own filter condition
                parameter_variables[parameter]: expressions[combination[parameter]]  # type: ignore[index]
                for parameter, expressions in enumerate(parameter_expressions)
                if combination[parameter] is not None
            }
            new_edge = JaniUnfolder.construct_edge(edge, substitution)
            new_edge.color = combination_coloring.get_or_make_color(combination)
            new_edges.append(new_edge)
        return new_edges

    @staticmethod
    def construct_edge(edge: Any, substitution: Any = None) -> Any:
        guard = stormpy.Expression(edge.template_edge.guard)
        assignments = edge.template_edge.assignments.clone()
        if substitution is not None:
            guard = guard.substitute(substitution)
            assignments.substitute(substitution, substitute_transcendental_numbers=True)
        template_edge = stormpy.storage.JaniTemplateEdge(guard)
        payntbind.synthesis.janiTemplateEdgeAddAssignments(template_edge, assignments)
        for dst in edge.template_edge.destinations:
            assignments = dst.assignments.clone()
            if substitution is not None:
                assignments.substitute(substitution, substitute_transcendental_numbers=True)
            template_edge.add_destination(stormpy.storage.JaniTemplateEdgeDestination(assignments))

        destinations = [(dst.target_location_index, dst.probability) for dst in edge.destinations]
        if substitution is not None:
            destinations = [(target, prob.substitute(substitution)) for target, prob in destinations]
        return stormpy.storage.JaniEdge(edge.source_location_index, edge.action_index, edge.rate, template_edge, destinations)

    def write_jani(self, output_path: str) -> None:
        logger.debug(f"Writing unfolded program to {output_path}")
        with open(output_path, "w") as f:
            f.write(str(self.jani_unfolded))
