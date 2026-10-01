from __future__ import annotations

from typing import Any

import stormpy

import paynt.colored_mdp
import paynt.synthesizer.conflict_generator.dtmc
import paynt.synthesizer.conflict_generator.mdp
import paynt.synthesizer.search_node
import paynt.task


class ConflictGeneratorAuto(paynt.synthesizer.conflict_generator.dtmc.ConflictGeneratorDtmc):
    """The DTMC conflict generator, for a model that is a Markov chain; for one that still has nondeterminism (the full assignment of an incomplete coloring,
    see ColoredMdp.build_assignment) the MDP conflict generator, which is set up only when such a model first turns up, so a coloring without any is not
    affected."""

    def __init__(self, colored_mdp: paynt.colored_mdp.ColoredMdp, task: paynt.task.SynthesisTask):
        super().__init__(colored_mdp, task)
        self.mdp_generator: paynt.synthesizer.conflict_generator.mdp.ConflictGeneratorMdp | None = None

    def construct_conflicts(
        self, node: paynt.synthesizer.search_node.SearchNode, assignment: Any, dtmc: Any, conflict_requests: list[tuple[int, Any, Any]]
    ) -> list[Any]:
        if isinstance(dtmc.model, (stormpy.storage.SparseDtmc, stormpy.storage.SparseExactDtmc)):
            return super().construct_conflicts(node, assignment, dtmc, conflict_requests)
        if self.mdp_generator is None:
            self.mdp_generator = paynt.synthesizer.conflict_generator.mdp.ConflictGeneratorMdp(self.colored_mdp, self.task)
            self.mdp_generator.initialize()
        return self.mdp_generator.construct_conflicts(node, assignment, dtmc, conflict_requests)
