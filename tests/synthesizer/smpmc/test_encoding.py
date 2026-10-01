"""ParameterBitVecEncoding: the Z3 variables of a parameter space, and the clauses and models SynthesizerSMPMC exchanges with Z3 through them."""

from __future__ import annotations

import itertools

import pytest
import z3

import paynt.parameter_space.parameter_space
import paynt.synthesizer.smpmc.encoding


def make_parameter_space(domains, names=None):
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for index, domain in enumerate(domains):
        parameter_space.add_parameter(names[index] if names else f"p{index}", list(domain))
    return parameter_space


def enumerate_models(encoding, extra_clauses):
    """Every assignment, as a tuple of options, that satisfies the domain of every parameter and extra_clauses."""
    solver = z3.Solver()
    for parameter in range(encoding.parameter_space.num_parameters):
        solver.add(z3.Or([encoding.variables[parameter] == encoding.value(option) for option in encoding.parameter_space.parameter_options(parameter)]))
    solver.add(*extra_clauses)
    models = set()
    while solver.check() == z3.sat:
        model = solver.model()
        values = tuple(model.eval(variable, model_completion=True).as_long() for variable in encoding.variables)
        models.add(values)
        solver.add(z3.Not(z3.And([variable == value for variable, value in zip(encoding.variables, values, strict=True)])))
    return models


@pytest.fixture
def encoding():
    return paynt.synthesizer.smpmc.encoding.ParameterBitVecEncoding(make_parameter_space([[0, 1], [0, 1, 2], [0, 1]]))


class TestVariables:
    def test_one_variable_per_parameter_of_one_width(self, encoding):
        assert len(encoding.variables) == 3
        assert {variable.size() for variable in encoding.variables} == {encoding.width}
        assert encoding.value(2).size() == encoding.width

    def test_equal_names_are_still_different_variables(self):
        """Parameter names are not unique (the POMDP and DT factories generate them), and Z3 interns variables by name."""
        encoding = paynt.synthesizer.smpmc.encoding.ParameterBitVecEncoding(make_parameter_space([[0, 1], [0, 1]], names=["x", "x"]))
        assert encoding.variables[0].get_id() != encoding.variables[1].get_id()
        assert sorted(encoding.name_to_parameter.values()) == [0, 1]
        for name, parameter in encoding.name_to_parameter.items():
            assert str(encoding.variables[parameter]) == name


class TestExclude:
    def full_assignment(self, encoding, options):
        assignment = encoding.parameter_space.copy()
        for parameter, option in enumerate(options):
            assignment.parameter_set_options(parameter, [option])
        return assignment

    def test_a_full_assignment_is_ruled_out_and_nothing_else(self, encoding):
        everything = set(itertools.product([0, 1], [0, 1, 2], [0, 1]))
        assert enumerate_models(encoding, []) == everything
        assert enumerate_models(encoding, [encoding.exclude(self.full_assignment(encoding, (1, 2, 0)))]) == everything - {(1, 2, 0)}

    def test_some_parameters_rule_out_every_assignment_that_agrees_on_them(self, encoding):
        everything = set(itertools.product([0, 1], [0, 1, 2], [0, 1]))
        clause = encoding.exclude(self.full_assignment(encoding, (1, 2, 0)), parameters=[0, 2])
        assert enumerate_models(encoding, [clause]) == {model for model in everything if (model[0], model[2]) != (1, 0)}

    def test_a_single_parameter(self, encoding):
        clause = encoding.exclude(self.full_assignment(encoding, (0, 1, 1)), parameters=[1])
        assert enumerate_models(encoding, [clause]) == {model for model in itertools.product([0, 1], [0, 1, 2], [0, 1]) if model[1] != 1}

    def test_an_excluded_parameter_has_to_be_fixed(self, encoding):
        with pytest.raises(AssertionError, match="every excluded parameter fixed"):
            encoding.exclude(encoding.parameter_space)


class TestExtractAssignment:
    @staticmethod
    def model_with(encoding, options):
        solver = z3.Solver()
        solver.add(*[variable == encoding.value(option) for variable, option in zip(encoding.variables, options, strict=True)])
        assert solver.check() == z3.sat
        return solver.model()

    def test_every_parameter_takes_its_value_in_the_model(self, encoding):
        assignment = encoding.extract_assignment(self.model_with(encoding, (1, 2, 0)), encoding.parameter_space)
        assert [assignment.parameter_options(parameter) for parameter in range(3)] == [[1], [2], [0]]

    def test_a_parameter_the_model_does_not_mention_gets_the_completion_value(self, encoding):
        solver = z3.Solver()
        assert solver.check() == z3.sat
        assignment = encoding.extract_assignment(solver.model(), encoding.parameter_space)
        assert [assignment.parameter_options(parameter) for parameter in range(3)] == [[0], [0], [0]]

    def test_free_parameters_keep_their_options(self, encoding):
        assignment = encoding.extract_assignment(self.model_with(encoding, (1, 2, 0)), encoding.parameter_space, free_parameters=[1])
        assert [assignment.parameter_options(parameter) for parameter in range(3)] == [[1], [0, 1, 2], [0]]

    def test_the_options_are_those_of_the_space_it_is_extracted_into(self, encoding):
        narrowed = encoding.parameter_space.copy()
        narrowed.parameter_set_options(1, [1, 2])
        assignment = encoding.extract_assignment(self.model_with(encoding, (0, 1, 1)), narrowed, free_parameters=[1])
        assert assignment.parameter_options(1) == [1, 2]
