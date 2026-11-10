from __future__ import annotations

from typing import Any

import stormpy
import payntbind


import os

import logging

logger = logging.getLogger(__name__)


class DrnParser:
    COMMENT_PREFIX = "//"
    TYPE_PREFIX = "@type: "
    STATE_PREFIX = "state "
    WHITESPACES = " \t\n\v\f\r"

    @classmethod
    def parse_drn(cls, sketch_path: str, use_exact: bool = False) -> Any:
        # try to read a drn file and return POSMG or POMDP based on the type
        # ValueError if file is not dnr or the model is of unsupported type
        explicit_model = None
        try:
            type = DrnParser.decide_type_of_drn(sketch_path)
            if type == "POSMG":
                if use_exact:
                    raise ValueError("Exact synthesis is not supported for POSMG models")
                pomdp_path = sketch_path + ".tmp"
                state_player_indications = DrnParser.pomdp_from_posmg(sketch_path, pomdp_path)
                pomdp = DrnParser.read_drn(pomdp_path)
                explicit_model = payntbind.synthesis.posmg_from_pomdp(pomdp, state_player_indications)
                os.remove(pomdp_path)
            else:
                explicit_model = DrnParser.read_drn(sketch_path, use_exact)
        except Exception as e:
            print(e)
            raise ValueError("Failed to read sketch file in a .drn format") from e
        return explicit_model

    @classmethod
    def decide_type_of_drn(cls, path: str) -> str:
        # decide type of model in drn file and return it as a string
        # path - path to drn file
        # return - string representation of type (e.g. 'POMDP')
        # ValueError if not valid drn file
        with open(path) as file:
            while True:
                line = file.readline()
                if line.isspace() or line.lstrip(cls.WHITESPACES).startswith(cls.COMMENT_PREFIX):
                    continue
                if line.startswith(cls.TYPE_PREFIX):
                    return line.removeprefix(cls.TYPE_PREFIX).removesuffix("\n")
                raise ValueError

    @classmethod
    def pomdp_from_posmg(cls, old_path: str, new_path: str) -> list[int]:
        # Change type of model in drn file from posmg to pomdp and store it in a new file
        # old_path - path to drn file with posmg model
        # new_path - path to drn file to store pomdp model
        # return - list indicating which states (indices) belongs to which player (values)
        posmg_file = open(old_path)
        pomdp_file = open(new_path, "w")
        state_player_indications = []

        for line in posmg_file:
            if line.startswith(cls.TYPE_PREFIX):
                line = line.replace("POSMG", "POMDP")
            if line.startswith(cls.STATE_PREFIX):
                l_idx = line.index("<")
                r_idx = line.index(">")
                player_num = line[l_idx + 1 : r_idx]
                state_player_indications.append(int(player_num))
                line = cls.str_remove_range(line, l_idx - 1, r_idx)  # -1 to remove the space before <

            line = line.replace(" []", "")  # temporaty fix
            pomdp_file.write(line)

        posmg_file.close()
        pomdp_file.close()

        return state_player_indications

    @staticmethod
    def str_remove_range(string: str, start_idx: int, end_idx: int) -> str:
        # return string without characters in specified range
        # both indices will be removed
        assert 0 <= start_idx < len(string)
        assert 0 <= end_idx < len(string)
        assert start_idx <= end_idx

        return string[:start_idx] + string[end_idx + 1 :]

    @classmethod
    def read_drn(cls, sketch_path: str, use_exact: bool = False) -> Any:
        builder_options = stormpy.io.DirectEncodingParserOptions()
        builder_options.build_choice_labeling = True
        return stormpy.build_model_from_drn[stormpy.Rational if use_exact else float](sketch_path, options=builder_options)

    @classmethod
    def write_model_in_pomdp_solve_format(cls, pomdp: Any, output_path: str, property_path: str) -> None:

        num_states = pomdp.nr_states
        num_obs = pomdp.nr_observations
        max_num_choices = max([pomdp.get_nr_available_actions(state) for state in range(num_states)])

        desc = f"""\
# auto-generated from PRISM program

discount: 0.95
values: reward
states: {num_states}
actions: {max_num_choices}
observations: {num_obs}

"""

        # initial state
        state_init = pomdp.initial_states[0]
        initial_distr: Any = [1 if state == state_init else 0 for state in range(num_states)]
        initial_distr = [str(x) for x in initial_distr]
        initial_distr = " ".join(initial_distr)
        desc += f"start:\n{initial_distr}\n\n"

        # transition matrix
        desc += "# transition matrix\n\n"

        # every action of every state, written out: the format needs a distribution for each, and the MADP parser takes no * for the action of T
        # a state with fewer actions repeats its last one, which adds no behaviour
        tm = pomdp.transition_matrix
        for state in range(num_states):
            group_start = tm.get_row_group_start(state)
            group_end = tm.get_row_group_end(state)
            for action_index in range(max_num_choices):
                row_index = min(group_start + action_index, group_end - 1)
                for entry in tm.get_row(row_index):
                    desc += f"T: {action_index} : {state} : {entry.column} {entry.value}\n"

        # observations
        desc += "\n# observations\n\n"
        for state in range(num_states):
            desc += f"O: * : {state} : {pomdp.observations[state]} 1\n"

        # rewards
        if len(pomdp.reward_models) > 0:
            # rewards
            desc += "\n# rewards\n\n"

            # assuming a single reward model
            rewards = next(iter(pomdp.reward_models.values()))

            # convert rewards to state-based
            state_rewards = []
            if rewards.has_state_rewards:
                state_rewards = list(rewards.state_rewards)
            elif rewards.has_state_action_rewards:
                state_action_rewards = list(rewards.state_action_rewards)
                for state in range(num_states):
                    group_start = tm.get_row_group_start(state)
                    state_rewards.append(state_action_rewards[group_start])
            else:
                raise TypeError("unknown reward type")

            # print state-based rewards
            for state in range(num_states):
                rew = state_rewards[state]
                if rew != 0:
                    desc += f"R: * : {state} : * : * {rew}\n"

        # ready to print
        logger.info(f"Writing POMDP in pomdp-solve format to {output_path} ...")
        with open(output_path, "w") as f:
            f.write(desc)
        logger.info(f"Writing default discount property to {property_path} ...")
        with open(property_path, "w") as f:
            f.write('R{"rew0"}max=? [F "target"]')
        logger.info("Write OK, aborting ...")
        exit(0)
