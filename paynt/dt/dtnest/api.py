import paynt.task

from .synthesizer import _run_dtnest
from .task import DtNestTask
from ..result import DtResult
from ..factory import DtColoredMdpFactory

__all__ = ["synthesize"]


def synthesize(
    cmdp_factory_dt: DtColoredMdpFactory,
    task: paynt.task.SynthesisTask,
    build_task: DtNestTask,
    depth_fine_tuning: bool = True,
    allow_perturbations: bool = True,
    recompute_scheduler_perturbation: bool = True,
) -> DtResult:
    """API function to solve a given SynthesisTask/DtNestTask pair against a DtColoredMdpFactory using the dtnest synthesizer.

    Returns paynt_result.
    """
    cmdp_factory_dt.build_task = build_task

    return _run_dtnest(
        cmdp_factory_dt,
        task,
        build_task.error_threshold,
        build_task.max_subtree_depth,
        depth_fine_tuning,
        allow_perturbations,
        recompute_scheduler_perturbation,
        task.timeout,
    )
