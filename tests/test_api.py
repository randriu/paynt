"""paynt.api.get_synthesizer: the synthesizer each kind of model gets (decision trees with SMPMC: tests/dt/test_smpmc_inner_engine.py)."""

import pytest

import paynt.api
import paynt.dt
import paynt.dt.dtnest
import paynt.mdp_family
import paynt.pomdp
import paynt.synthesizer.synthesizer_cegis
import paynt.synthesizer.synthesizer_onebyone

from tests.helpers.models import load_sketch

CASES = [
    ("tests/smpmc-tiny", "sketch.templ", "cegis", False, False, paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS),
    ("tests/pomdp-maze", "sketch.templ", "ar", True, False, paynt.pomdp.PomdpSynthesizer),
    ("tests/decpomdp-dectiger", "dectiger.dpomdp", "ar", True, False, paynt.pomdp.decpomdp.DecPomdpSynthesizer),
    ("tests/posmg-test-game", "sketch.templ", "ar", True, False, paynt.pomdp.posmg.PosmgSynthesizer),
    ("tests/mdp-family-rocks-4-2", "sketch.templ", "onebyone", False, False, paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne),
    ("tests/mdp-family-rocks-4-2", "sketch.templ", "ar", False, False, paynt.mdp_family.PolicyTreeSynthesizer),
    ("tests/dt-orchard", "sketch.templ", "ar", False, False, paynt.dt.DtSynthesizer),
    ("tests/dt-orchard", "sketch.templ", "ar", False, True, paynt.dt.dtnest.DtNest),
]


@pytest.mark.parametrize(("project", "sketch_name", "method", "fsc_synthesis", "dtnest", "expected"), CASES)
def test_each_kind_of_model_gets_its_synthesizer(project, sketch_name, method, fsc_synthesis, dtnest, expected):
    factory, task = load_sketch(project, sketch_name=sketch_name)
    synthesizer = paynt.api.get_synthesizer(factory, task, method, fsc_synthesis=fsc_synthesis, dtnest=dtnest)
    assert type(synthesizer) is expected
    assert synthesizer.task is task


def test_a_pomdp_family_has_nothing_to_synthesize():
    factory, task = load_sketch("tests/pomdp-family-avoid-smaller")
    with pytest.raises(SystemExit) as exit:
        paynt.api.get_synthesizer(factory, task, "ar")
    assert exit.value.code == 0
