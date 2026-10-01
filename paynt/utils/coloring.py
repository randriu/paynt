"""Telling apart the two ways a colored MDP's choices can be colored: by explicit (parameter, option) pairs, as in a payntbind Coloring, or by an arbitrary
formula, as in a ColoringGeneral (see paynt.utils.coloring_builder), which has no pairs to read back.

Where the pairs are needed, see
paynt.utils.error_handling.require_pair_list_coloring.
"""

from __future__ import annotations

from typing import Any

import payntbind.synthesis


def is_general_coloring(coloring: Any) -> bool:
    """Whether coloring is a payntbind ColoringGeneral rather than a coloring given by explicit (parameter, option) pairs."""
    return isinstance(coloring, payntbind.synthesis.ColoringGeneral)
