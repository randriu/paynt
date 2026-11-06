"""paynt.pomdp._utils: the Q-values of an FSC and the belief update, on pomdp-maze with one memory node."""

import pytest

import paynt.pomdp
import paynt.pomdp._utils


@pytest.fixture
def maze_fsc(pomdp_maze):
    """(the synthesizer, the assignment of the best FSC)"""
    synthesizer = paynt.pomdp.PomdpSynthesizer(pomdp_maze.factory, pomdp_maze.task, method="ar")
    assignment = synthesizer.synthesize(synthesizer.colored_mdp.parameter_space, print_stats=False)
    return synthesizer, assignment


def test_the_q_value_of_the_initial_state_is_the_value_of_the_fsc(maze_fsc):
    synthesizer, assignment = maze_fsc
    qvalues = paynt.pomdp._utils.compute_qvalues(synthesizer.colored_mdp, assignment, synthesizer.task.specification)
    initial = synthesizer.colored_mdp.feature_info.pomdp.initial_states[0]
    assert qvalues[initial][0] == pytest.approx(synthesizer.task.specification.optimality.optimum, rel=1e-4)


def test_the_belief_after_an_observation_follows_bayes_rule(pomdp_maze):
    """The first action places the agent at random; the belief after seeing an observation is the placement restricted to its states."""
    info = pomdp_maze.colored_mdp.feature_info
    pomdp = info.pomdp
    initial = pomdp.initial_states[0]
    [action] = info.action_labels_at_observation[pomdp.observations[initial]]
    row = {entry.column: entry.value for entry in pomdp.transition_matrix.get_row(pomdp.get_choice_index(initial, 0))}
    for observation in {pomdp.observations[state] for state in row}:
        seen = {state: probability for state, probability in row.items() if pomdp.observations[state] == observation}
        belief = paynt.pomdp._utils.next_belief(info, {initial: 1.0}, action, observation)
        assert belief == pytest.approx({state: probability / sum(seen.values()) for state, probability in seen.items()})
