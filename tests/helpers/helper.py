import os
import stormpy
import stormpy.examples
import stormpy.examples.files

import payntbind.synthesis

import paynt.colored_mdp
import paynt.parser.sketch
from paynt.examples import paynt_models_dir

stormpy_example_dir = stormpy.examples.files.testfile_dir


def get_stormpy_example_path(*paths):
    return os.path.join(stormpy_example_dir, *paths)


def get_sketch_paths(project_path, sketch_name="sketch.templ", props_name="sketch.props"):
    return os.path.join(paynt_models_dir, project_path, sketch_name), os.path.join(paynt_models_dir, project_path, props_name)


def load_colored_mdp(project_path, props_name="sketch.props"):
    """Load a fixture's (ColoredMdp, SynthesisTask) pair directly, for tests that build several colorings/synthesizers from the same project and so need
    independently-loaded ColoredMdp/task pairs rather than a single shared fixture (Synthesizer instances mutate task.specification state)."""
    sketch_path, props_path = get_sketch_paths(project_path, props_name=props_name)
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path)
    return factory.build(), task


def general_colored_mdp(colored_mdp):
    """Build a ColoredMdp sharing colored_mdp's underlying MDP/parameter space, but with its Coloring lifted into a ColoringGeneral via fromColoring --
    feature_kind/feature_info carried over unchanged, since build_assignment's DTMC-vs-MDP branch (and any feature-specific code) keys off feature_kind, not the
    coloring's own type."""
    general = payntbind.synthesis.ColoringGeneral.fromColoring(
        colored_mdp.parameter_space.native, colored_mdp.underlying_mdp.nondeterministic_choice_indices, colored_mdp.coloring
    )
    result = paynt.colored_mdp.ColoredMdp(
        colored_mdp.underlying_mdp, colored_mdp.parameter_space, general, colored_mdp.use_exact, feature_kind=colored_mdp.feature_kind
    )
    result.feature_info = colored_mdp.feature_info
    return result
