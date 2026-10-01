from __future__ import annotations

from typing import Any

from . import version

import paynt.api
import paynt.utils.timer
import paynt.utils.version_check
import paynt.parser.sketch

import paynt.pomdp
import paynt.dt._cli
import paynt.dt.dtnest._cli
import paynt.pomdp._cli
import paynt.pomdp.saynt._cli
import paynt.mdp_family._cli
import paynt.synthesizer.smpmc._cli

import rich_click as click
import sys
import os
import cProfile
import pstats

import logging

logger = logging.getLogger(__name__)


def add_options(options: list[Any]) -> Any:
    """Standard click idiom for composing a decorator list built elsewhere (here, a feature's own _cli.py) onto a command function, applied in the same order as
    if the decorators had been written inline."""

    def _add_options(func: Any) -> Any:
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Eager --version callback: prints the version and exits immediately, before click validates the otherwise-required PROJECT argument."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"Paynt {version()}")
    ctx.exit()


def setup_logger(log_path: str | None = None) -> list[logging.Handler]:
    """Setup routine for logging."""
    root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(message)s")

    handlers: list[logging.Handler] = []
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        handlers.append(fh)
    sh = logging.StreamHandler(sys.stdout)
    handlers.append(sh)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    for h in handlers:
        root.addHandler(h)
    return handlers


@click.command()
@click.argument("project", type=click.Path(exists=True))
@click.option("--sketch", default="sketch.templ", show_default=True, panel="Model & specification loading", help="name of the sketch file in the project")
@click.option("--props", default="sketch.props", show_default=True, panel="Model & specification loading", help="name of the properties file in the project")
@click.option(
    "--constraint-bound",
    type=click.FLOAT,
    panel="Model & specification loading",
    help="bound for creating constrained POMDP for Cassandra models",
)
@click.option("--relative-error", type=click.FLOAT, default="0", show_default=True, panel="Synthesis", help="relative error for optimal synthesis")
@click.option("--optimum-threshold", type=click.FLOAT, panel="Synthesis", help="known optimum bound")
@click.option("--precision", type=click.FLOAT, default=1e-4, panel="Synthesis", help="model checking precision")
@click.option("--exact", is_flag=True, default=False, panel="Synthesis", help="use exact synthesis (very limited at the moment)")
@click.option(
    "--sound",
    is_flag=True,
    default=False,
    panel="Synthesis",
    help="do not cap value/policy iteration: never uses an unconverged model checking result, but a slowly converging solve may run indefinitely",
)
@click.option("--timeout", type=int, panel="Synthesis", help="timeout (s)")
@click.option(
    "--method", type=click.Choice(["onebyone", "ar", "cegis", "hybrid", "smpmc"]), default="ar", show_default=True, panel="Synthesis", help="synthesis method"
)
@click.option("--disable-expected-visits", is_flag=True, default=False, panel="Synthesis", help="do not compute expected visits for the splitting heuristic")
@click.option(
    "--ce-generator",
    type=click.Choice(["dtmc", "mdp"]),
    default="dtmc",
    show_default=True,
    panel="Synthesis",
    help="counterexample generator",
)
@click.option(
    "--constraint",
    type=click.Choice(["exists", "exists_forall", "costs", "prob0", "prob1"]),
    default=None,
    panel="Synthesis",
    help="custom constraint over the parameter space (SMPMC and CEGIS); defaults to a plain existential search",
)
@click.option(
    "--costs-threshold", type=int, default=None, panel="Synthesis", help="threshold for --constraint costs (reads a sketch.costs file in the project)"
)
@click.option(
    "--fsc-synthesis",
    is_flag=True,
    default=False,
    panel="FSC synthesis (POMDP / Dec-POMDP / POSMG / family)",
    help="enable incremental synthesis of FSCs for a (Dec-)POMDP",
)
@click.option(
    "--fsc-memory-size",
    default=1,
    show_default=True,
    panel="FSC synthesis (POMDP / Dec-POMDP / POSMG / family)",
    help="implicit memory size for (Dec-)POMDP FSCs",
)
@add_options(paynt.pomdp._cli.options)
@add_options(paynt.pomdp.saynt._cli.options)
@add_options(paynt.mdp_family._cli.options)
@add_options(paynt.dt._cli.options)
@add_options(paynt.dt.dtnest._cli.options)
@add_options(paynt.synthesizer.smpmc._cli.options)
@click.option("--export", type=click.Choice(["jani", "drn", "pomdp"]), panel="Output", help="export the model to specified format and abort")
@click.option("--export-synthesis", type=click.Path(), default=None, panel="Output", help="base filename to output synthesis result")
@click.option("--profiling", is_flag=True, default=False, panel="Output", help="run profiling")
@click.option(
    "--version", is_flag=True, default=False, is_eager=True, expose_value=False, callback=print_version, panel="Output", help="print PAYNT version and exit"
)
def paynt_run(
    project: str,
    sketch: str,
    props: str,
    relative_error: float,
    optimum_threshold: float | None,
    precision: float,
    exact: bool,
    sound: bool,
    timeout: int | None,
    export: str | None,
    method: str,
    disable_expected_visits: bool,
    fsc_synthesis: bool,
    fsc_memory_size: int,
    posterior_aware: bool,
    storm_pomdp: bool,
    iterative_storm: tuple[int, int, int] | None,
    get_storm_result: int | None,
    storm_options: str,
    prune_storm: bool,
    use_storm_cutoffs: bool,
    unfold_strategy_storm: str,
    export_synthesis: str | None,
    mdp_discard_unreachable_choices: bool,
    tree_depth: int,
    tree_enumeration: bool,
    tree_map_scheduler: str | None,
    add_dont_care_action: bool,
    constraint_bound: float | None,
    dtnest: bool,
    dtnest_subtree_depth: int,
    dtnest_error_threshold: float,
    ce_generator: str,
    constraint: str | None,
    costs_threshold: int | None,
    smpmc_forall: str | None,
    smpmc_verify_robust: bool,
    profiling: bool,
) -> None:

    profiler = None
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    paynt.utils.timer.GlobalTimer.start(timeout)

    logger.info(f"This is Paynt version {version()}.")
    paynt.utils.version_check.check_stormpy_compatibility()

    # Every option below that affects synthesis behavior (as opposed to model loading/parsing) is threaded
    # through as a SynthesisTask/feature-task field. Sketch.load_sketch doesn't know the sketch's feature
    # until it has parsed it, so task_kwargs carries every feature's options at once; both the generic
    # SynthesisTask and whichever feature-specific task ends up being constructed pick out only the keys
    # they recognize (see paynt.task.SynthesisTask.from_specification and e.g. paynt.dt.task.DtTask).
    task_kwargs: dict[str, Any] = {
        "export_synthesis_filename_base": export_synthesis,
        "conflict_generator_type": ce_generator,
        "disable_expected_visits": disable_expected_visits,
        "memory_size": fsc_memory_size,
        "posterior_aware": posterior_aware,
        "discard_unreachable_choices": mdp_discard_unreachable_choices,
        "tree_depth": tree_depth,
        "tree_enumeration": tree_enumeration,
        "scheduler_path": tree_map_scheduler,
        "add_dont_care_action": add_dont_care_action,
        "max_subtree_depth": dtnest_subtree_depth,
        "error_threshold": dtnest_error_threshold,
        "constraint_name": constraint,
        "costs_threshold": costs_threshold,
        "costs_file_path": os.path.join(project, "sketch.costs"),
        "forall_pattern": smpmc_forall,
        "verify_robust": smpmc_verify_robust,
    }

    storm_control = None
    if storm_pomdp:
        storm_control = paynt.pomdp.saynt.StormPOMDPControl()
        storm_control.set_options(storm_options, get_storm_result, iterative_storm, use_storm_cutoffs, unfold_strategy_storm, prune_storm)

    sketch_path = os.path.join(project, sketch)
    properties_path = os.path.join(project, props)
    colored_mdp_factory, task = paynt.parser.sketch.Sketch.load_sketch(
        sketch_path, properties_path, export, relative_error, precision, constraint_bound, exact, task_kwargs=task_kwargs, sound=sound
    )
    synthesizer = paynt.api.get_synthesizer(colored_mdp_factory, task, method, fsc_synthesis, storm_control, dtnest)
    synthesizer.run(optimum_threshold)

    if profiling:
        assert profiler is not None
        profiler.disable()
        print_profiler_stats(profiler)


def print_profiler_stats(profiler: cProfile.Profile) -> None:
    stats = pstats.Stats(profiler)
    NUM_LINES = 10

    logger.debug("cProfiler info:")
    stats.sort_stats("tottime").print_stats(NUM_LINES)

    logger.debug("percentage breakdown:")
    # .stats/.total_tt are genuine pstats.Stats attributes at runtime, just missing from typeshed's stub
    entries = [(key, data[2]) for key, data in stats.stats.items()]  # type: ignore[attr-defined]
    entries = sorted(entries, key=lambda x: x[1], reverse=True)
    entries = entries[:NUM_LINES]
    for key, data in entries:
        module, line, method = key
        if module == "~":
            callee = method
        else:
            callee = f"{module}:{line}({method})"
        percentage = round(data / stats.total_tt * 100, 1)  # type: ignore[attr-defined]
        percentage_str = str(percentage).ljust(4)
        print(f"{percentage_str} %  {callee}")


def main() -> None:
    setup_logger()
    paynt_run()


if __name__ == "__main__":
    main()
