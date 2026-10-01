"""The Z3 theory-solver plugin: a z3.UserPropagateBase subclass that turns Z3's own CDCL search into the CDCL(T) integration described in Section 4.2-4.3 of
arXiv:2511.08078.

Ported from molehill's plugins/search.py:SearchMarkovChain (https://github.com/linusheck/molehill,
GPL-3.0), with one correctness fix found empirically during this port and worth calling out explicitly:

Z3's preprocessor eagerly constant-folds a `viable(...)` argument into a numeral whenever that argument
is pinned by a simple equality -- not only for a literal constant written in the source formula, but for
*any* variable Z3 can already prove has exactly one possible value. In particular, every single-option
parameter (a hole whose in_range clause collapses to one equality) triggers this on the very first
`created` callback, for every query. An argument handled this way never appears in a `fixed` callback at
all, since it was never a live variable to begin with -- so a naive implementation that only tracks
*variable* arguments (as molehill's own bare-name-keyed dict implicitly does, and as an early draft of
this module did) silently drops that parameter from every `fixed` dict it builds. Silently, because the
solver still finds a model -- just one the theory was never actually asked about, since the theory's own
completeness argument depends on `fixed` reflecting every relevant parameter, constant-folded or not.
The fix is to track each `viable(...)` argument by its *position* (which is also its parameter index,
since it is always called as viable(*variables_in_parameter_index_order), see synthesizer.py) and record
constant-folded arguments directly, rather than only arguments that later arrive via `fixed`.
"""

from __future__ import annotations

from typing import Any

import z3

import paynt.synthesizer.smpmc.checker

import logging

logger = logging.getLogger(__name__)


class SmpmcPropagator(z3.UserPropagateBase):
    def __init__(
        self,
        solver: Any,
        ctx: Any,
        theory: paynt.synthesizer.smpmc.checker.ColoredMdpTheory,
        name_to_parameter: dict[str, int],
        owner_ctx: Any = None,
    ):
        super().__init__(solver, ctx)
        self.add_fixed(self._fixed)
        self.add_final(self._final)
        self.add_created(self._created)

        self.theory = theory
        self.name_to_parameter = name_to_parameter

        # ast key -> current value (bool for a viable(...) literal, int for a parameter variable)
        self.partial_model: dict[str, Any] = {}
        # undo log: keys pushed onto partial_model, in order, so pop() can unwind exactly num_scopes levels
        self.trail: list[str] = []
        # trail marks, one per push()
        self.scopes: list[int] = []
        # per scope: the theory state (see _theory_state) once that push's analysis found no conflict, None otherwise
        self.scope_states: list[tuple[int, bool, int] | None] = []
        # ast key of a viable(...) application -> its argument specs, positional (position == parameter
        # index): ("const", value) for a Z3-constant-folded argument, ("var", arg_key) for a live one
        self.viable_args: dict[str, list[tuple[str, Any]]] = {}
        # ast key -> the actual z3 term, needed to build conflict() dependencies
        self.term_of: dict[str, Any] = {}
        # the context every term in term_of is held through: the solver's own, also in a fresh() propagator. Z3 frees
        # an MBQI sub-context once its round ends, but z3's Python API keeps every propagator registered until
        # interpreter exit, so a term held through the sub-context would be released through a dangling context
        # pointer, corrupting the heap. Sub-contexts share the solver's AST manager, so the solver's context can hold
        # their terms just as well.
        self.owner_ctx = owner_ctx if owner_ctx is not None else solver.ctx

    @staticmethod
    def _key(ast: Any) -> str:
        """A cheap, stable key for a Z3 AST node -- the Z3 APIs for structural identity are too slow to call on this hot path (molehill's own comment on the
        equivalent code), so this uses the printed s-expression for an application (distinguishing e.g. different viable(...) argument tuples) and the
        declaration name for a plain variable."""
        text = ast.sexpr()
        return text if text.startswith("(") else ast.decl().name()

    def _retain(self, term: Any) -> Any:
        return z3.ExprRef(term.as_ast(), self.owner_ctx)

    def _created(self, term: Any) -> None:
        key = self._key(term)
        args: list[tuple[str, Any]] = []
        for i in range(term.num_args()):
            arg = term.arg(i)
            if z3.Z3_is_numeral_ast(arg.ctx_ref(), arg.as_ast()):
                args.append(("const", arg.as_long()))
            else:
                arg_key = self._key(arg)
                self.term_of[arg_key] = self._retain(arg)
                self.add(arg)
                args.append(("var", arg_key))
        self.viable_args[key] = args
        self.term_of[key] = self._retain(term)

    def _fixed(self, ast: Any, value: Any) -> None:
        key = self._key(ast)
        if key not in self.term_of:
            self.term_of[key] = self._retain(ast)
        if z3.is_true(value):
            decoded: Any = True
        elif z3.is_false(value):
            decoded = False
        else:
            decoded = value.as_long()
        self.partial_model[key] = decoded
        self.trail.append(key)

    def _theory_state(self) -> tuple[int, bool, int]:
        """Changes whenever the theory may answer a query differently: after each Storm call (whose verdict it caches), once a first full assignment has been
        checked, and when the optimality threshold tightens."""
        return (self.theory.mc_calls, self.theory.first_full_assignment_checked, self.theory.epoch)

    def _interrupt_if_time_is_up(self) -> bool:
        """Z3 calls back at every decision, which is a far better clock than its own timeout parameter (that fires late).

        Interrupting here is safe, as a check() is running (an interrupt sent while none is leaves the context cancelled).

        :returns: whether the search was interrupted
        """
        if not self.theory.time_is_up():
            return False
        z3.Z3_interrupt(self.ctx_ref())
        if self.ctx_ref().value != self.owner_ctx.ref().value:
            # a fresh() propagator, run by an MBQI round in a sub-context: stop the search that started the round as well
            z3.Z3_interrupt(self.owner_ctx.ref())
        return True

    def push(self) -> None:
        if self._interrupt_if_time_is_up():
            self.scopes.append(len(self.trail))
            self.scope_states.append(None)
            return
        mark = len(self.trail)
        # Nothing fixed since the enclosing push, whose analysis found no conflict, and the theory unchanged since, also by
        # the fresh() propagators sharing it: analysing again would ask the same queries and get the same answers.
        # molehill skips on the first condition alone, which also skips pruning newly enabled by the shared theory.
        redundant = bool(self.scopes) and self.scopes[-1] == mark and self.scope_states[-1] == self._theory_state()
        self.scopes.append(mark)
        if redundant:
            self.scope_states.append(self._theory_state())
            return
        conflict = self._analyse()
        self.scope_states.append(None if conflict else self._theory_state())

    def pop(self, num_scopes: int) -> None:
        for _ in range(num_scopes):
            mark = self.scopes.pop()
            self.scope_states.pop()
            while len(self.trail) > mark:
                key = self.trail.pop()
                del self.partial_model[key]

    def _final(self) -> None:
        if not self._interrupt_if_time_is_up():
            self._analyse()

    def _analyse(self) -> bool:
        """Check every currently-fixed viable(...) literal against the theory; on the first refutation, push a learned conflict clause back into Z3 and stop (Z3
        will call back in for more once it has backtracked past this conflict).

        :returns: whether a conflict was pushed

        Deliberately does NOT wait for every argument of viable(...) to be fixed before consulting the
        theory -- Algorithm 1 in the paper is explicit that it works over whatever partial model K
        currently holds ("for all literals k in K do ... eta <- {theta | theta satisfies K \\ {v}}"), not
        just a complete one. An earlier version of this method required every parameter to be fixed first,
        which is a correctness-preserving but severely performance-destroying simplification: since
        viable(...) is asserted over every parameter, "wait for all of them" means the theory is *never*
        consulted until Z3 has already committed to one fully concrete candidate, degenerating the whole
        search into one-candidate-at-a-time enumeration with no early pruning from partial information --
        exactly the CEGIS-like behavior SMPMC is meant to avoid, confirmed empirically (see the
        conversation this was found in: a controlled comparison on a small synthetic instance showed a
        4x reduction in total theory calls -- most of them genuinely partial -- once this requirement was
        removed, with identical sat/unsat answers either way). Parameters not yet fixed are simply left
        out of `fixed`; checker.py's eta construction already treats an unmentioned parameter as "still
        fully open" and reasons about the induced sub-MDP accordingly.

        One exception, ported from molehill's own Mole.partial_model_consistent (its "magic trick" that
        checks a DTMC first): a *partial* (non-empty, not-fully-fixed) query is skipped entirely --
        reported inconclusive without ever calling the theory -- until at least one fully-fixed assignment
        has been checked. This isn't in tension with the paragraph above; it's a different failure mode
        the partial-consultation fix doesn't address on its own. Partial consultation pays off when a
        partial query can be *refuted*, pruning a whole region for free -- but on a property that's easy
        to satisfy (found on a real 1.65M-state DTMC-with-holes model, arXiv:2511.08078's own benchmark
        suite: a reward threshold so generous that virtually every completion satisfies it), no partial
        query is ever refutable, so every one of them is pure, expensive overhead: an 8-second VI call on
        a barely-smaller sub-MDP, paid once per parameter Z3 fixes on the way to its first candidate, for
        no pruning benefit at all. Deferring partial checks until Z3 has found *some* concrete candidate
        first sidesteps this without giving up early pruning altogether -- it only delays it until pruning
        has a chance to actually pay for itself.
        """
        for key, value in list(self.partial_model.items()):
            if not isinstance(value, bool):
                continue
            arg_specs = self.viable_args.get(key)
            if arg_specs is None:
                continue

            fixed: dict[int, int] = {}
            for parameter, (kind, payload) in enumerate(arg_specs):
                if kind == "const":
                    fixed[parameter] = payload
                elif payload in self.partial_model:
                    fixed[parameter] = self.partial_model[payload]
                # else: this parameter isn't fixed yet -- leave it out of `fixed`, don't abort the analysis

            is_full_assignment = len(fixed) == len(arg_specs)
            if not is_full_assignment and fixed and not self.theory.first_full_assignment_checked:
                continue

            try:
                result = self.theory.check(fixed, value)
            except Exception as error:
                # z3py only prints what a callback raises and goes on as if the theory had found nothing wrong -- accepting an assignment that was never
                # checked. Keep the error, stop the search, and let the synthesizer raise it.
                self.theory.failure = error
                z3.Z3_interrupt(self.ctx_ref())
                return False
            if is_full_assignment:
                self.theory.first_full_assignment_checked = True
            if result is None:
                continue

            deps = [self.term_of[key]]
            for parameter in result.conflict_parameters:
                kind, payload = arg_specs[parameter]
                if kind == "var":
                    deps.append(self.term_of[payload])
            self.conflict(deps=deps)
            return True
        return False

    def fresh(self, new_ctx: Any) -> SmpmcPropagator:
        """Called by Z3 to create a sub-context propagator, notably during MBQI quantifier instantiation for robust synthesis (Section 4.4).

        Shares theory (hence its cache) with the parent -- MBQI's sub-contexts explore the same colored MDP, so there is no reason to lose cached results, and
        the theory's own epoch/cache state must stay consistent across every propagator instance in a run.
        """
        return SmpmcPropagator(None, new_ctx, self.theory, self.name_to_parameter, owner_ctx=self.owner_ctx)
