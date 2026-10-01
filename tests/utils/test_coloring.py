"""paynt.utils.coloring: telling a ColoringGeneral from a coloring given by explicit (parameter, option) pairs.

The function takes the raw payntbind coloring, so it is exercised on the two lifted fixtures' colorings the way the consumers use it.
"""

from __future__ import annotations

import pytest

import paynt.utils.coloring

from helpers.helper import general_colored_mdp, load_colored_mdp

FIXTURES = ["tests/smpmc-tiny", "tests/mdp-family-avoid-8-2-easy"]


@pytest.mark.parametrize("project", FIXTURES)
class TestIsGeneralColoring:
    def test_false_for_a_standard_coloring_and_true_once_lifted(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        assert paynt.utils.coloring.is_general_coloring(colored_mdp.coloring) is False
        assert paynt.utils.coloring.is_general_coloring(general_colored_mdp(colored_mdp).coloring) is True

    def test_is_what_the_colored_mdp_property_reports(self, project):
        colored_mdp, _task = load_colored_mdp(project)
        for candidate in (colored_mdp, general_colored_mdp(colored_mdp)):
            assert candidate.has_general_coloring is paynt.utils.coloring.is_general_coloring(candidate.coloring)
