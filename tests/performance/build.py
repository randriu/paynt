"""Build the colored MDP of a sketch without synthesizing anything."""

import argparse
import os

import paynt.parser.sketch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project")
parser.add_argument("--sketch", default="sketch.templ")
parser.add_argument("--props", default="sketch.props")
parser.add_argument("--fsc-memory-size", type=int, default=1)
args = parser.parse_args()

factory, task = paynt.parser.sketch.Sketch.load_sketch(
    os.path.join(args.project, args.sketch), os.path.join(args.project, args.props), task_kwargs={"memory_size": args.fsc_memory_size}
)
colored_mdp = factory.build()
mdp = colored_mdp.underlying_mdp
print(f"built: {mdp.nr_states} states, {mdp.nr_choices} choices, {colored_mdp.parameter_space.num_parameters} parameters")
