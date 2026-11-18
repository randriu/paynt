"""The command line run in this process, through click's test runner: a synthesis run, and the exports read back and compared with the model PAYNT builds."""

from __future__ import annotations

import collections
import os
import re
import shutil

import pytest
import stormpy
from click.testing import CliRunner

import paynt
import paynt.cli
import paynt.parser.sketch
import paynt.utils.timer
from paynt.examples import paynt_models_dir

from tests.helpers.models import load_colored_mdp, load_sketch


@pytest.fixture
def run_paynt(monkeypatch, tmp_path):
    """Runs the command line on a copy of a project of models/ and returns the folder of the copy and what was printed."""
    # the command line starts the global timer
    monkeypatch.setattr(paynt.utils.timer.GlobalTimer, "global_timer", None)

    def run(project, *args):
        folder = tmp_path / os.path.basename(project)
        shutil.copytree(os.path.join(paynt_models_dir, project), folder)
        result = CliRunner().invoke(paynt.cli.paynt_run, [str(folder), *args])
        assert result.exit_code == 0, repr(result.exception)
        return folder, result.output

    return run


def test_a_run_logs_the_optimum_and_a_design_that_attains_it(run_paynt, caplog):
    """At most 4 hours, the least energy: 4 units, with the trips of 4 hours."""
    caplog.set_level("INFO")
    run_paynt("tests/time-energy", "--props", "optimality.props", "--method", "cegis")
    assert "optimum: 4.0" in caplog.text
    assert any(f"FIRST={first}, SECOND={second}" in caplog.messages for first, second in [(0, 2), (1, 1), (2, 0)])


def test_the_drn_export_is_the_model_paynt_builds(run_paynt):
    folder, _output = run_paynt("tests/smpmc-tiny", "--export", "drn")
    exported = stormpy.build_model_from_drn(str(folder / "sketch.drn"))
    built = load_colored_mdp("tests/smpmc-tiny")[0].underlying_mdp
    assert str(exported.transition_matrix) == str(built.transition_matrix)
    for label in exported.labeling.get_labels():
        assert exported.labeling.get_states(label) == built.labeling.get_states(label)
    assert list(exported.reward_models) == list(built.reward_models)


def test_the_jani_export_is_the_unfolded_family(run_paynt):
    folder, _output = run_paynt("tests/smpmc-tiny", "--export", "jani")
    jani, _properties = stormpy.parse_jani_model(str(folder / "sketch.jani"))
    exported = stormpy.build_model(jani)
    built = load_colored_mdp("tests/smpmc-tiny")[0].underlying_mdp
    assert str(exported.transition_matrix) == str(built.transition_matrix)


def read_pomdp_solve(path):
    """The rows of every state, as sorted (successor, probability) tuples, one per action, the observation of every state and the number of actions."""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    observations = {}
    with open(path) as file:
        lines = file.read().splitlines()
    num_actions = int(next(line for line in lines if line.startswith("actions:")).split()[1])
    for line in lines:
        if line.startswith("T:"):
            _, action, state, entry = (part.strip() for part in line.split(":"))
            successor, probability = entry.split()
            rows[int(state)][int(action)].append((int(successor), float(probability)))
        if line.startswith("O:"):
            _, _action, state, entry = (part.strip() for part in line.split(":"))
            observations[int(state)] = int(entry.split()[0])
    return {state: sorted(tuple(sorted(row)) for row in actions.values()) for state, actions in rows.items()}, observations, num_actions


def test_the_pomdp_solve_export_has_the_transitions_and_observations_of_the_pomdp(run_paynt):
    """A state with fewer actions than the file repeats its last one, so only the distinct rows are compared there."""
    folder, _output = run_paynt("tests/pomdp-maze", "--export", "pomdp")
    rows, observations, num_actions = read_pomdp_solve(folder / "sketch.pomdp")
    pomdp = load_sketch("tests/pomdp-maze")[0].pomdp
    matrix = pomdp.transition_matrix
    for state in range(pomdp.nr_states):
        expected = sorted(
            tuple(sorted((entry.column, entry.value) for entry in matrix.get_row(row)))
            for row in range(matrix.get_row_group_start(state), matrix.get_row_group_end(state))
        )
        assert len(rows[state]) == num_actions
        assert sorted(set(rows[state])) == sorted(set(expected))
        assert observations[state] == pomdp.observations[state]


def test_paynt_reads_its_pomdp_solve_export_back(run_paynt):
    folder, _output = run_paynt("tests/pomdp-maze", "--export", "pomdp")
    factory, _task = paynt.parser.sketch.Sketch.load_sketch(str(folder / "sketch.pomdp"), str(folder / "props.pomdp"))
    # the 15 states and an initial one the Cassandra parser adds
    assert factory.pomdp.nr_states == 16


def test_version_needs_no_project():
    result = CliRunner().invoke(paynt.cli.paynt_run, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"Paynt {paynt.version()}\n"


def test_profiling_prints_the_share_of_the_most_expensive_calls(run_paynt):
    _folder, output = run_paynt("tests/smpmc-tiny", "--method", "onebyone", "--profiling")
    lines = [line for line in output.splitlines() if re.match(r"\d+\.\d\s* %  \S", line)]
    shares = [float(line.split("%")[0]) for line in lines]
    assert 0 < len(shares) <= 10
    # the time spent in each call itself, so together at most the whole run (up to rounding)
    assert shares == sorted(shares, reverse=True)
    assert 0 < shares[0] and sum(shares) <= 100.5
    # a Python function by its file and line, a built-in by its name
    assert any(re.search(r"\.py:\d+\(\w+\)$", line) for line in lines)
    assert any("<built-in method" in line for line in lines)
    assert not any("~:" in line or "None" in line for line in lines)


def test_conditional_parameters_per_phase_give_a_copy_of_x_for_after_the_condition(run_paynt, caplog):
    """In models/tests/conditional, x=0 first and then x=1 after the condition always reaches the goal."""
    caplog.set_level("INFO")
    run_paynt("tests/conditional", "--conditional-parameters", "per-phase", "--conditional-algorithm", "restart")
    assert "optimum: 1.0" in caplog.text
    assert "x=0, x@condition=1" in caplog.messages
