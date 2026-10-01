from __future__ import annotations

from typing import Any

import paynt.task

from .synthesizer import _choose_solver_for_dt_task, _run_dt_map_scheduler, _run_dtpaynt
from .task import DtTask
from .result import DtResult
from .factory import DtColoredMdpFactory

__all__ = ["synthesize", "create_task", "create_colored_mdp_factory"]


def synthesize(
    cmdp_factory_dt: DtColoredMdpFactory,
    task: paynt.task.SynthesisTask,
    build_task: DtTask,
    use_solver: str | None = None,
    method: str = "ar",
) -> DtResult:
    """API function to solve a given SynthesisTask/DtTask pair against a DtColoredMdpFactory.

    Optional use_solver parameter can force a specific solver to be used. Returns paynt_result.

    :param method: "ar" or "smpmc", see DtSynthesizer; mapping a scheduler supports only "ar"
    """
    cmdp_factory_dt.build_task = build_task

    if use_solver is None:
        use_solver = _choose_solver_for_dt_task(build_task)

    assert use_solver in ["dtmap", "dtpaynt"], f"Invalid solver choice: {use_solver}. Valid options are 'dtmap' and 'dtpaynt'."

    if use_solver == "dtmap":
        return _run_dt_map_scheduler(cmdp_factory_dt, task, build_task.scheduler_to_map, build_task.tree_depth, method)
    return _run_dtpaynt(cmdp_factory_dt, task, build_task.tree_depth, task.timeout, method)


def create_task(properties: list[Any], tree_depth: int) -> DtTask:
    """API function to create a DtTask from a list of StormPy properties and a tree depth."""
    raise NotImplementedError("API not yet implemented.")


def create_colored_mdp_factory(model: Any) -> DtColoredMdpFactory:
    """API function to create a DtColoredMdpFactory from a StormPy model."""
    raise NotImplementedError("API not yet implemented.")
