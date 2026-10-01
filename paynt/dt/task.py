from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(kw_only=True)
class DtTask:
    """Feature-specific build knobs for decision-tree synthesis, owned by DtColoredMdpFactory (factory.build_task)."""

    # the depth of the tree to synthesize (CLI's --tree-depth)
    tree_depth: int = 0
    tree_enumeration: bool = False
    # path to a JSON scheduler file to be mapped to a decision tree (CLI's --tree-map-scheduler)
    scheduler_path: str | None = None
    # if true, an explicit action executing a random choice of an available action will be added to
    # each state (consumed by DtColoredMdpFactory at construction time)
    add_dont_care_action: bool = True
    # an already-in-memory scheduler to map (distinct from scheduler_path above, which is a file path
    # loaded by the CLI's own run loop) -- the library-facing entry point used by paynt.dt.api.synthesize.
    # init=False: never a real constructor argument, only ever set later via set_scheduler_to_map.
    scheduler_to_map: Any = field(init=False, default=None)

    def set_scheduler_to_map(self, scheduler: Any) -> None:
        self.scheduler_to_map = scheduler

    @property
    def has_scheduler_to_map(self) -> bool:
        return self.scheduler_to_map is not None
