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
    """The SynthesisTask sibling of smpmc_tiny_colored_mdp_factory -- loaded independently (see tests/pomdp/conftest.py's equivalent fixtures) so each fixture
    has its own SynthesisTask, not one threaded through the factory."""
    sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny")
    _factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def smpmc_tiny_colored_mdp(smpmc_tiny_colored_mdp_factory):
    return smpmc_tiny_colored_mdp_factory.build()


@pytest.fixture
def smpmc_tiny_optimality_colored_mdp_factory():
    sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny", props_name="optimality.props")
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def smpmc_tiny_optimality_task():
    sketch_path, props_path = get_sketch_paths("tests/smpmc-tiny", props_name="optimality.props")
    _factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def smpmc_tiny_optimality_colored_mdp(smpmc_tiny_optimality_colored_mdp_factory):
    return smpmc_tiny_optimality_colored_mdp_factory.build()


@pytest.fixture
def mdp_family_colored_mdp_factory():
    sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory


@pytest.fixture
def mdp_family_task():
    sketch_path, props_path = get_sketch_paths("tests/mdp-family-avoid-8-2-easy")
    _factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def mdp_family_colored_mdp(mdp_family_colored_mdp_factory):
    return mdp_family_colored_mdp_factory.build()


@pytest.fixture
def rocks_task():
    sketch_path, props_path = get_sketch_paths("tests/mdp-family-rocks-4-2")
    _factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return task


@pytest.fixture
def rocks_colored_mdp():
    sketch_path, props_path = get_sketch_paths("tests/mdp-family-rocks-4-2")
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory.build()
