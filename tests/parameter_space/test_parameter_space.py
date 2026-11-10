"""ParameterSpace: splitting the options of a parameter."""

import paynt.parameter_space.parameter_space


def test_suboptions_unique_gives_each_used_option_a_part_of_its_own():
    space = paynt.parameter_space.parameter_space.ParameterSpace()
    space.add_parameter("p", [str(option) for option in range(7)])
    parts = space.suboptions_unique(0, [5, 2])
    assert [part[0] for part in parts] == [5, 2]
    assert sorted(option for part in parts for option in part) == list(range(7))
    assert all(len({5, 2} & set(part)) == 1 for part in parts)
