"""The same runs give the same answers: in two processes with different hash seeds, so that no engine depends on the order of a set or a dict, and again in the
same process.

Hybrid is left out, as it switches between AR and CEGIS on timing.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_RUN_SCRIPT = textwrap.dedent("""
    import paynt.synthesizer.synthesizer
    from tests.helpers.models import load_colored_mdp

    cases = [("tests/smpmc-tiny", "sketch.props"), ("tests/smpmc-tiny", "optimality.props"), ("tests/time-energy", "sketch.props")]
    for project, props in cases:
        for method in ["onebyone", "ar", "cegis", "smpmc"]:
            colored_mdp, task = load_colored_mdp(project, props_name=props)
            if method == "smpmc" and task.specification.num_properties > 1:
                continue
            result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
            print("ANSWER", project, props, method, result.assignment, result.value)
    """)


_REPEAT_SCRIPT = textwrap.dedent("""
    import paynt.synthesizer.synthesizer
    from tests.helpers.models import load_colored_mdp

    cases = {"onebyone": "tests/time-energy", "ar": "tests/time-energy", "cegis": "tests/time-energy", "smpmc": "tests/smpmc-tiny"}
    for method, project in cases.items():
        for _ in range(6):
            colored_mdp, task = load_colored_mdp(project)
            result = paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp, task, method).run()
            print("ANSWER", method, result.assignment)
    """)


def answers(script, hash_seed=0):
    repository = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    environment = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=repository, env=environment, timeout=300)
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.startswith("ANSWER")]


def test_two_processes_give_the_same_answers():
    first = answers(_RUN_SCRIPT, hash_seed=1)
    assert len(first) == 11
    assert answers(_RUN_SCRIPT, hash_seed=2) == first


@pytest.fixture(scope="module")
def repeated_runs():
    runs = {}
    for line in answers(_REPEAT_SCRIPT):
        _, method, assignment = line.split(" ", 2)
        runs.setdefault(method, []).append(assignment)
    return runs


@pytest.mark.parametrize(
    "method",
    [
        "onebyone",
        "ar",
        pytest.param("cegis", marks=pytest.mark.xfail(strict=True, reason="the SMT solver is in Z3's global context, whose models change from call to call")),
        "smpmc",
    ],
)
def test_the_same_run_six_times_in_one_process_gives_the_same_answer(repeated_runs, method):
    assert len(repeated_runs[method]) == 6
    assert len(set(repeated_runs[method])) == 1
