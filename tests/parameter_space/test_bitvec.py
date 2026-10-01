import z3

import paynt.parameter_space.bitvec
import paynt.parameter_space.parameter_space


def _parameter_space(*option_counts):
    parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
    for i, count in enumerate(option_counts):
        parameter_space.add_parameter(f"p{i}", [str(o) for o in range(count)])
    return parameter_space


class TestBitvecWidth:
    def test_width_covers_the_largest_parameter(self):
        # bitvec_width computes ceil(log2(max_options + 1)) + 1; for 4 options that's ceil(log2(5))+1 = 4
        parameter_space = _parameter_space(2, 4, 3)
        assert paynt.parameter_space.bitvec.bitvec_width(parameter_space) == 4

    def test_single_option_parameter_still_gets_a_usable_width(self):
        parameter_space = _parameter_space(1)
        width = paynt.parameter_space.bitvec.bitvec_width(parameter_space)
        assert width >= 1

    def test_no_parameters_does_not_crash(self):
        parameter_space = _parameter_space()
        assert paynt.parameter_space.bitvec.bitvec_width(parameter_space) == 1


class TestParameterBitvecVariables:
    def test_one_variable_per_parameter(self):
        parameter_space = _parameter_space(2, 4)
        variables, width, name_to_parameter = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)
        assert len(variables) == 2
        assert all(isinstance(v, z3.BitVecRef) for v in variables)
        assert all(v.size() == width for v in variables)

    def test_variables_are_named_by_index_not_bare_name(self):
        """Two parameters sharing a name must not collide into the same Z3 variable -- see parameter_bitvec_variables' docstring."""
        parameter_space = paynt.parameter_space.parameter_space.ParameterSpace()
        parameter_space.add_parameter("dup", ["a", "b"])
        parameter_space.add_parameter("dup", ["c", "d"])
        variables, _width, name_to_parameter = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)
        assert str(variables[0]) != str(variables[1])
        assert name_to_parameter[str(variables[0])] == 0
        assert name_to_parameter[str(variables[1])] == 1

    def test_name_to_parameter_round_trips(self):
        parameter_space = _parameter_space(3, 5, 2)
        variables, _width, name_to_parameter = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)
        for parameter, variable in enumerate(variables):
            assert name_to_parameter[str(variable)] == parameter
