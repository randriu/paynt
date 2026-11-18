from __future__ import annotations

from typing import Any

import stormpy
import payntbind

import paynt.parameter_space.parameter_space
import paynt.specification.property
import paynt.parser.jani
import paynt.model.model_builder

import os
import re
import tempfile

import logging

logger = logging.getLogger(__name__)


class PrismParser:
    @classmethod
    def read_prism(
        cls, sketch_path: str, properties_path: str, relative_error: float, use_exact: bool = False
    ) -> tuple[Any, Any, paynt.specification.property.Specification, paynt.parameter_space.parameter_space.ParameterSpace | None, Any, Any, Any]:

        # parse the program
        prism, parameter_definitions = PrismParser.load_sketch_prism(sketch_path)
        expression_parser = stormpy.expressions.ExpressionParser(prism.expression_manager)
        expression_parser.set_identifier_mapping({})
        prism_model_type = {
            stormpy.io.PrismModelType.DTMC: "DTMC",
            stormpy.io.PrismModelType.MDP: "MDP",
            stormpy.io.PrismModelType.POMDP: "POMDP",
        }[prism.model_type]
        logger.debug("PRISM model type: " + prism_model_type)

        # parse constants

        # parse parameter definitions
        parameter_expressions = None
        parameter_space = None
        if len(parameter_definitions) > 0:
            logger.info("processing parameter definitions...")
            prism, parameter_expressions, parameter_space = PrismParser.parse_parameters(prism, expression_parser, parameter_definitions)
        prism = prism.label_unlabelled_commands({})

        specification = PrismParser.parse_specification(properties_path, relative_error, prism, use_exact=use_exact)

        # construct the underlying model
        coloring = None
        jani_unfolder = None
        obs_evaluator = None
        if parameter_space is not None:
            assert prism_model_type in ["DTMC", "MDP", "POMDP"], "parameter detected, but the program is neither DTMC nor (PO)MDP"
            assert parameter_expressions is not None
            # unfold parameter options via Jani
            jani_unfolder = paynt.parser.jani.JaniUnfolder(prism, parameter_expressions, specification, parameter_space, use_exact=use_exact)
            specification = jani_unfolder.specification
            underlying_mdp = jani_unfolder.underlying_mdp
            coloring = payntbind.synthesis.Coloring(
                parameter_space.native, underlying_mdp.nondeterministic_choice_indices, jani_unfolder.choice_to_parameter_options
            )
            if prism.model_type == stormpy.io.PrismModelType.POMDP:
                obs_evaluator = payntbind.synthesis.ObservationEvaluator(prism, underlying_mdp)
            underlying_mdp = payntbind.synthesis.addChoiceLabelsFromJani(underlying_mdp)
        else:
            underlying_mdp = paynt.model.model_builder.ModelBuilder.from_prism(prism, specification, use_exact)

        return prism, underlying_mdp, specification, parameter_space, coloring, jani_unfolder, obs_evaluator

    @classmethod
    def load_sketch_prism(cls, sketch_path: str) -> tuple[Any, list[tuple[str, str, str]]]:
        # read lines
        with open(sketch_path) as f:
            sketch_lines = f.readlines()

        # replace "hole" declarations (sketch template syntax) with constants
        hole_re_brace = re.compile(r"^\s*hole\s+(.*?)\s+(.*?)\s+in\s+\{(.*?)\}\s*;")
        # hole_re_bracket = re.compile(r'^\s*hole\s+(.*?)\s+(.*?)\s+in\s+[(.*?)]\s+;')
        sketch_output = []
        parameter_definitions = []
        observables_line = False
        for line in sketch_lines:
            # Treat observation definition via "observables" keyword
            if line.startswith("observables"):
                observables_line = True
                line = line.split("//")[0].strip()  # remove comments
                if len(line) > len("observables"):
                    line = line[len("observables") :].strip()
                else:
                    continue
            if observables_line:
                if line.startswith("endobservables"):
                    observables_line = False
                    continue
                line = line.split("//")[0].strip()  # remove comments
                observables = line.strip().split(",")
                observables = [obs.strip() for obs in observables]
                for obs in observables:
                    sketch_output.append(f'observable "{obs}" = {obs};\n')
                continue

            match = hole_re_brace.search(line)
            if match is None:
                sketch_output.append(line)
                continue
            parameter_type = match.group(1)
            parameter_name = match.group(2)
            parameter_options = match.group(3).replace(" ", "")
            parameter_definitions.append((parameter_name, parameter_type, parameter_options))
            sketch_output.append(f"const {parameter_type} {parameter_name};\n")
            sketch_output.append(f"const {parameter_type} {parameter_name}_MIN;\n")
            sketch_output.append(f"const {parameter_type} {parameter_name}_MAX;\n")

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = os.path.join(directory, os.path.basename(sketch_path))
            with open(tmp_path, "w") as f:
                f.writelines(sketch_output)
            try:
                prism = stormpy.parse_prism_program(tmp_path, prism_compatibility=True)
            except Exception as e:
                raise SyntaxError from e

        return prism, parameter_definitions

    @classmethod
    def parse_parameters(
        cls, prism: Any, expression_parser: Any, parameter_definitions: list[tuple[str, str, str]]
    ) -> tuple[Any, list[list[Any]], paynt.parameter_space.parameter_space.ParameterSpace]:

        # parse parameter definitions
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        parameter_expressions = []
        parameter_min = []
        parameter_max = []
        for parameter_name, parameter_type, parameter_options in parameter_definitions:
            if ".." in parameter_options:
                assert parameter_type == "int" or parameter_type == "double", "cannot use range-based definitions for non-integer of non-double parameter types"
                if parameter_type == "double":
                    assert ":" in parameter_options, (
                        "using range-based definition for double requires specifying the increment step range_start..range_end:step"
                    )
                    range_start = float(parameter_options[0 : parameter_options.find("..")])
                    range_end = float(parameter_options[parameter_options.find("..") + 2 : parameter_options.find(":")].strip())
                    increment_string = parameter_options.split(":")[-1].strip()
                    increment = float(increment_string)
                    increment_decimal_precision = len(increment_string.split(".")[-1])
                    parameter_min.append(range_start)
                    parameter_max.append(range_end)
                    steps = (range_end - range_start) / increment
                    options = ["{:.{n}f}".format(range_start + x * increment, n=increment_decimal_precision) for x in range(int(round(steps) + 1))]
                    if float(options[-1]) > range_end:
                        options = options[:-1]
                else:
                    if ":" in parameter_options:
                        range_start = int(parameter_options[0 : parameter_options.find("..")])
                        range_end = int(parameter_options[parameter_options.find("..") + 2 : parameter_options.find(":")].strip())
                        increment = int(parameter_options.split(":")[-1].strip())
                    else:
                        options = parameter_options.split("..")
                        range_start = int(options[0])
                        range_end = int(options[1])
                        increment = 1
                    parameter_min.append(range_start)
                    parameter_max.append(range_end)
                    options = [str(o) for o in range(range_start, range_end + 1, increment)]
            else:
                options = parameter_options.split(",")
                options_numerical: list[float]
                if parameter_type == "int":
                    options_numerical = [int(o) for o in options]
                else:
                    options_numerical = [float(o) for o in options]
                parameter_min.append(min(options_numerical))
                parameter_max.append(max(options_numerical))
            expressions = [expression_parser.parse_from_string(o) for o in options]
            parameter_expressions.append(expressions)

            option_labels = [str(e) for e in expressions]
            parameter_space.add_parameter(parameter_name, option_labels)

        # substitute constants used as min/max values of parameters
        parameter_range_definitions = {}
        for parameter in range(parameter_space.num_parameters):
            parameter_name = parameter_space.parameter_name(parameter)
            var_min = prism.get_constant(f"{parameter_name}_MIN").expression_variable
            parameter_range_definitions[var_min] = expression_parser.parse_from_string(str(parameter_min[parameter]))
            var_max = prism.get_constant(f"{parameter_name}_MAX").expression_variable
            parameter_range_definitions[var_max] = expression_parser.parse_from_string(str(parameter_max[parameter]))
        prism = prism.define_undefined_constants(parameter_range_definitions)

        # check that all undefined constants are indeed the parameters
        parameter_names = [parameter_space.parameter_name(parameter) for parameter in range(parameter_space.num_parameters)]
        for c in prism.constants:
            if not c.defined:
                assert c.name in parameter_names, f"constant {c.name} was not specified"

        return prism, parameter_expressions, parameter_space

    @classmethod
    def parse_property(cls, line: str, prism: Any = None) -> Any:
        """Parse a line containing a single PCTL property.

        @return the property or None if no property was detected
        """
        if prism is not None:
            props = stormpy.parse_properties(line, prism)
        else:
            props = stormpy.parse_properties(line)
        if len(props) == 0:
            return None
        if len(props) > 1:
            logger.warning("multiple properties detected on one line, dropping all but the first one")
        return props[0]

    @classmethod
    def parse_specification(
        cls, properties_path: str, relative_error: float = 0, prism: Any = None, use_exact: bool = False
    ) -> paynt.specification.property.Specification:
        """Expecting one property per line.

        The line may be terminated with a semicolon. Empty lines or comments are allowed.
        """
        if not os.path.isfile(properties_path):
            raise ValueError(f"the properties file {properties_path} does not exist")
        logger.info(f"loading properties from {properties_path} ...")

        with open(properties_path) as file:
            lines = list(file)

        properties = []

        for line in lines:
            formula = PrismParser.parse_property(line, prism)
            if formula is None:
                continue
            prop = paynt.specification.property.construct_property(formula, relative_error, use_exact)
            properties.append(prop)

        specification = paynt.specification.property.Specification(properties)
        logger.info(f"found the following specification: {specification}")
        return specification
