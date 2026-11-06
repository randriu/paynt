"""The test models of models/tests, one fixture each, named after its folder.

See tests/README.md for what each model is.
"""

import pytest

from tests.helpers.models import Project


@pytest.fixture
def smpmc_tiny():
    return Project("tests/smpmc-tiny")


@pytest.fixture
def smpmc_tiny_optimality():
    return Project("tests/smpmc-tiny", props_name="optimality.props")


@pytest.fixture
def mdp_family_avoid_8_2_easy():
    return Project("tests/mdp-family-avoid-8-2-easy")


@pytest.fixture
def mdp_family_rocks_4_2():
    return Project("tests/mdp-family-rocks-4-2")


@pytest.fixture
def pomdp_family_avoid_smaller():
    return Project("tests/pomdp-family-avoid-smaller")


@pytest.fixture
def pomdp_maze():
    return Project("tests/pomdp-maze")


@pytest.fixture
def decpomdp_dectiger():
    # no properties file: the discounted reward the model itself declares
    return Project("tests/decpomdp-dectiger", sketch_name="dectiger.dpomdp")


@pytest.fixture
def posmg_mec_test():
    return Project("tests/posmg-mec-test")


@pytest.fixture
def posmg_test_game():
    return Project("tests/posmg-test-game")


@pytest.fixture
def dt_orchard():
    return Project("tests/dt-orchard")
