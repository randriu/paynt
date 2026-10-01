from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paynt.dt.task import DtTask


@dataclass(kw_only=True)
class DtNestTask(DtTask):
    """The settings of a dtnest synthesis, on top of those of DtTask.

    The depth of the tree to synthesize, DtTask.tree_depth, is left as it is: dtnest does not change it. What dtnest has is the depth of the subtrees, which
    stays inside its iteration (see max_subtree_depth).
    """

    error_threshold: float = 0.05
    initial_tree: Any = None
    # The depth of the subtrees that the iteration of dtnest works with
    max_subtree_depth: int = 7
