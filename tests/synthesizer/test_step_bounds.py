"""Step-bounded properties (F<=k, U<=k, X, C<=k, I=k): Storm gives no scheduler for them on an MDP, so what needs one refuses them."""

import pytest

import paynt.api
import paynt.parser.sketch

from tests.helpers.models import get_sketch_paths


def load(tmp_path, project, sketch_name, prop_text, **task_kwargs):
    props_path = tmp_path / "case.props"
    props_path.write_text(prop_text + "\n")
    return paynt.parser.sketch.Sketch.load_sketch(get_sketch_paths(project, sketch_name)[0], str(props_path), task_kwargs=task_kwargs)


class TestDesignFamily:
    """until.templ: within one step only H=1 reaches the goal, with probability 1/2."""

    @pytest.mark.parametrize("method, engine", [("ar", "AR"), ("cegis", "CEGIS"), ("hybrid", "CEGIS")])
    def test_ar_and_cegis_refuse_it(self, tmp_path, method, engine):
        factory, task = load(tmp_path, "tests/edge-cases", "until.templ", 'Pmax=? [F<=1 "goal"]')
        with pytest.raises(NotImplementedError, match=f"^{engine} .*does not support step-bounded properties.*use --method onebyone or smpmc"):
            paynt.api.get_synthesizer(factory, task, method)

    @pytest.mark.parametrize("method", ["onebyone", "smpmc"])
    def test_onebyone_and_smpmc_check_it(self, tmp_path, method):
        factory, task = load(tmp_path, "tests/edge-cases", "until.templ", 'Pmax=? [F<=1 "goal"]')
        result = paynt.api.get_synthesizer(factory, task, method).run()
        assert str(result.assignment) == "H=1"
        assert result.value == pytest.approx(0.5)


class TestMdpFamily:
    """step-bounded-family.templ: within two steps a=0 reaches the goal with probability 0.9, by a policy that depends on the step."""

    @pytest.mark.parametrize("prop_text, refusal", [('Pmax=? [F<=2 "goal"]', "step-bounded"), ('R{"steps"}min=? [F "goal"]', "only probability")])
    def test_the_policy_tree_refuses_it_and_rewards(self, tmp_path, prop_text, refusal):
        factory, task = load(tmp_path, "tests/edge-cases", "step-bounded-family.templ", prop_text)
        with pytest.raises(NotImplementedError, match=f"^the policy tree .*{refusal}"):
            paynt.api.get_synthesizer(factory, task, "ar")

    @pytest.mark.parametrize("method", ["onebyone", "smpmc"])
    def test_onebyone_and_smpmc_report_no_policy(self, tmp_path, method, caplog):
        factory, task = load(tmp_path, "tests/edge-cases", "step-bounded-family.templ", 'Pmax=? [F<=2 "goal"]')
        result = paynt.api.get_synthesizer(factory, task, method).run()
        assert str(result.assignment) == "a=0"
        assert result.value == pytest.approx(0.9)
        assert result.selected_choices is None
        assert "no policy is reported for a=0" in caplog.text

    def test_verifying_a_robust_result_refuses_it(self, tmp_path):
        factory, task = load(
            tmp_path, "tests/edge-cases", "step-bounded-family.templ", 'P>=0.5 [F<=2 "goal"]', constraint_name="exists_forall", verify_robust=True
        )
        with pytest.raises(NotImplementedError, match="^--smpmc-verify-robust does not support step-bounded properties.*drop the option"):
            paynt.api.get_synthesizer(factory, task, "smpmc")


class TestDecisionTree:
    @pytest.mark.parametrize("dtnest, refusal", [(False, "decision-tree synthesis with --method ar"), (True, "--dtnest")])
    def test_ar_refuses_it(self, tmp_path, dtnest, refusal):
        factory, task = load(tmp_path, "tests/dt-maze", "sketch.templ", "Pmax=? [F<=5 goal]", tree_depth=1)
        with pytest.raises(NotImplementedError, match=f"^{refusal} does not support step-bounded properties.*--method smpmc"):
            paynt.api.get_synthesizer(factory, task, "ar", dtnest=dtnest)

    def test_smpmc_checks_it(self, tmp_path):
        factory, task = load(tmp_path, "tests/dt-maze", "sketch.templ", "Pmax=? [F<=5 goal]", tree_depth=1)
        assert paynt.api.get_synthesizer(factory, task, "smpmc").run().value == pytest.approx(0.236264, abs=1e-6)
