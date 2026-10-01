import pytest

import paynt.parser.sketch

from helpers.helper import get_sketch_paths


@pytest.fixture
def smpmc_tiny_colored_mdp_factory():
    sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def smpmc_tiny_task():
    sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
    _factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def smpmc_tiny_colored_mdp(smpmc_tiny_colored_mdp_factory):
    return smpmc_tiny_colored_mdp_factory.build()
