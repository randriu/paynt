"""
api - "User-friendly" API for PAYNT.

This module exposes high-level functions for programmatic use of PAYNT.
"""

from __future__ import annotations

from typing import Any

from . import version

__all__ = ["get_version", "get_synthesizer"]


def get_version() -> str:
    """Return PAYNT version string."""
    return version()


def get_synthesizer(
    colored_mdp_factory: Any,
    task: Any,
    method: str = "ar",
    fsc_synthesis: bool = False,
    storm_control: Any = None,
    dtnest: bool = False,
) -> Any:
    """Get the correct synthesizer for the given colored MDP and task.

    :param colored_mdp_factory: the factory that produces the ColoredMdp to synthesize
    :param task: the SynthesisTask this run is solving
    :param method: the generic algorithm to fall back on when no feature-specific driver applies
        ("onebyone"/"ar"/"cegis"/"hybrid")
    :param fsc_synthesis: for FSC-unfolding features (POMDP/POSMG/Dec-POMDP), enable incremental FSC
        synthesis over increasing memory sizes rather than plain synthesis over the current unfolding
    :param storm_control: for POMDP with fsc_synthesis, an optional StormPOMDPControl to run SAYNT instead
        of plain PAYNT POMDP synthesis
    :param dtnest: for decision trees, use the dtnest synthesizer instead of plain AR
    :return: a synthesizer ready to .run()/.synthesize()/.evaluate()
    """
    import paynt.synthesizer.synthesizer
    import paynt.utils.error_handling

    feature_kind = colored_mdp_factory.feature_kind

    if feature_kind == "pomdp_family":
        # a family-of-POMDPs sketch isn't run through a Synthesizer at all (see e.g.
        # paynt.mdp_family.pomdp._utils.build_dtmc_sketch instead)
        import logging

        logging.getLogger(__name__).info("nothing to do with the POMDP sketch, aborting...")
        exit(0)

    if feature_kind == "dt":
        from paynt.dt import DtSynthesizer
        from paynt.dt.dtnest import DtNest

        if dtnest:
            paynt.utils.error_handling.require_method_ar(method, "--dtnest")
            return DtNest(colored_mdp_factory, task)
        return DtSynthesizer(colored_mdp_factory, task, method=method)

    if feature_kind == "pomdp" and fsc_synthesis:
        import paynt.pomdp

        if storm_control is not None:
            return paynt.pomdp.saynt.SayntSynthesizer(colored_mdp_factory, task, method, storm_control)
        return paynt.pomdp.PomdpSynthesizer(colored_mdp_factory, task, method)

    if feature_kind == "decpomdp" and fsc_synthesis:
        import paynt.pomdp

        return paynt.pomdp.decpomdp.DecPomdpSynthesizer(colored_mdp_factory, task, method)

    if feature_kind == "family":
        if method in ("onebyone", "smpmc"):
            return paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp_factory.build(), task, method)
        import paynt.mdp_family

        return paynt.mdp_family.PolicyTreeSynthesizer(colored_mdp_factory.build(), task)

    if feature_kind == "posmg" and fsc_synthesis:
        import paynt.pomdp

        return paynt.pomdp.posmg.PosmgSynthesizer(colored_mdp_factory, task, method)

    return paynt.synthesizer.synthesizer.Synthesizer.for_method(colored_mdp_factory.build(), task, method)
