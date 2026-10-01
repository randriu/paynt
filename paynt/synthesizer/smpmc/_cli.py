"""SMPMC CLI options, composed into paynt.cli.paynt_run via add_options([...])."""

import rich_click as click

_PANEL = "SMPMC"

options = [
    click.option(
        "--smpmc-forall",
        type=str,
        default=None,
        panel=_PANEL,
        help="regex over parameter names selecting the universally quantified parameters of --constraint exists_forall; "
        "optional for an MDP family, where it defaults to the family's own parameters (the environment)",
    ),
    click.option(
        "--smpmc-verify-robust",
        is_flag=True,
        default=False,
        panel=_PANEL,
        help="independently verify a robust (--constraint exists_forall) result with AR on the negated specification",
    ),
]
