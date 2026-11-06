"""Loading the models of models/ in tests."""

import functools
import os

import payntbind.synthesis

import paynt.colored_mdp
import paynt.parser.sketch
from paynt.examples import paynt_models_dir


def get_sketch_paths(project, sketch_name="sketch.templ", props_name="sketch.props"):
    return os.path.join(paynt_models_dir, project, sketch_name), os.path.join(paynt_models_dir, project, props_name)


def load_sketch(project, sketch_name="sketch.templ", props_name="sketch.props", **kwargs):
    """The (factory, task) pair of a project of models/, the keyword arguments go to Sketch.load_sketch."""
    return paynt.parser.sketch.Sketch.load_sketch(*get_sketch_paths(project, sketch_name, props_name), **kwargs)


def load_colored_mdp(project, sketch_name="sketch.templ", props_name="sketch.props", **kwargs):
    """The (colored_mdp, task) pair of a project of models/."""
    factory, task = load_sketch(project, sketch_name, props_name, **kwargs)
    return factory.build(), task


class Project:
    """A project of models/ parsed once: its factory, its task and the colored MDP the factory builds."""

    def __init__(self, project, sketch_name="sketch.templ", props_name="sketch.props"):
        self.factory, self.task = load_sketch(project, sketch_name, props_name)

    @functools.cached_property
    def colored_mdp(self):
        return self.factory.build()


def general_colored_mdp(colored_mdp):
    """The colored MDP with its coloring lifted into a ColoringGeneral, sharing the MDP and the parameter space."""
    general = payntbind.synthesis.ColoringGeneral.fromColoring(
        colored_mdp.parameter_space.native, colored_mdp.underlying_mdp.nondeterministic_choice_indices, colored_mdp.coloring
    )
    result = paynt.colored_mdp.ColoredMdp(
        colored_mdp.underlying_mdp, colored_mdp.parameter_space, general, colored_mdp.use_exact, feature_kind=colored_mdp.feature_kind
    )
    result.feature_info = colored_mdp.feature_info
    return result
