"""Every engine on a tiny sketch, its answer compared with the reference of tests/helpers/reference.py."""

from __future__ import annotations

import math

import paynt.parser.sketch
import paynt.synthesizer.synthesizer

from tests.helpers.models import general_colored_mdp
from tests.helpers.reference import Reference

ENGINES = ["onebyone", "ar", "cegis", "hybrid", "smpmc", "onebyone, general", "cegis, general", "smpmc, general"]
# the default model checking precision
PRECISION = 1e-4


def close(a, b):
    return math.isclose(a, b, rel_tol=PRECISION, abs_tol=PRECISION)


def refuses(engine, reference):
    """Whether the engine does not support the specification: CEGIS (also in hybrid) a maximized reward, SMPMC several properties."""
    method = engine.split(",")[0]
    if method in ("cegis", "hybrid"):
        return any(prop.reward and prop.maximizing for prop in reference.properties)
    return method == "smpmc" and len(reference.properties) > 1


def wrong_answer(reference, result, relative_error=0):
    """What is wrong with the result, in a few words, or None if it is right."""
    if not reference.feasible:
        return None if not result.success else f"finds {result.assignment}"
    if not result.success:
        return "says infeasible"
    design = reference.design(result.assignment)
    if design is None:
        return f"finds {result.assignment}, which has a deadlock"
    if not reference.admissible(design):
        return f"finds {design}, which is not admissible"
    if reference.optimum is None:
        return None
    value, optimum = float(design.values[reference.objective]), float(reference.optimum)
    if not close(result.value, value):
        return f"reports {result.value:.6g} for {design}, whose value is {value:.6g}"
    if reference.properties[reference.objective].maximizing:
        good_enough = value * (1 + relative_error) >= optimum - PRECISION
    else:
        good_enough = value * (1 - relative_error) <= optimum + PRECISION
    return None if good_enough else f"finds {design} with {value:.6g}, the optimum is {optimum:.6g}"


def answer(engine, sketch_path, props_path, reference, relative_error=0):
    """None if the engine answers right or refuses as it should, else what it does wrong."""
    method, _, coloring = engine.partition(", ")
    factory, task = paynt.parser.sketch.Sketch.load_sketch(sketch_path, props_path, relative_error=relative_error)
    colored_mdp = factory.build()
    if coloring == "general":
        colored_mdp = general_colored_mdp(colored_mdp)
    try:
        result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
    except Exception as error:
        if isinstance(error, NotImplementedError) and refuses(engine, reference):
            return None
        return f"raises {type(error).__name__}: {str(error).splitlines()[0]}"
    if refuses(engine, reference):
        return "does not refuse"
    return wrong_answer(reference, result, relative_error)


def wrong_answers(sketch_path, properties, directory, relative_error=0):
    """The engines that answer wrong, with what they do wrong.

    :param directory: where to write the properties file
    """
    reference = Reference(sketch_path, properties)
    props_path = directory / "case.props"
    props_path.write_text("\n".join(properties) + "\n")
    return {engine: problem for engine in ENGINES if (problem := answer(engine, str(sketch_path), str(props_path), reference, relative_error)) is not None}
