from __future__ import annotations

from typing import Any, TYPE_CHECKING

import z3

import paynt.parameter_space.bitvec
import paynt.parameter_space.constraints.constraint
import paynt.parameter_space.parameter_space

if TYPE_CHECKING:
    # search_node imports this module, so only import it for annotations to avoid a circular import at
    # runtime -- from __future__ import annotations means these hints are never evaluated eagerly anyway
    import paynt.colored_mdp
    import paynt.synthesizer.search_node
    import paynt.task

import logging

logger = logging.getLogger(__name__)


class ParameterSpaceEncoding:
    def __init__(self, smt_solver: SmtSolver, parameter_space: paynt.parameter_space.parameter_space.ParameterSpace):

        self.smt_solver = smt_solver
        self.parameter_space = parameter_space

        # for each parameter, a formula encoding its possible options
        self.parameter_clauses: list[Any] = []
        # SMT formula describing the parameter_space
        self.encoding: Any = None
        # set to False as soon as pick_assignment returns None
        self.has_assignments = True

        parameter_clauses = []
        for parameter in range(parameter_space.num_parameters):
            all_clauses = smt_solver.solver_clauses[parameter]
            clauses = [all_clauses[option] for option in parameter_space.parameter_options(parameter)]
            or_clause = clauses[0] if len(clauses) == 1 else z3.Or(clauses)
            parameter_clauses.append(or_clause)

        encoding = parameter_clauses[0] if len(parameter_clauses) == 1 else z3.And(parameter_clauses)
        if smt_solver.constraint_clauses:
            encoding = z3.And(encoding, *smt_solver.constraint_clauses)

        self.parameter_clauses = parameter_clauses
        self.encoding = encoding

    def pick_assignment(self) -> paynt.parameter_space.parameter_space.ParameterSpace | None:

        if not self.has_assignments:
            return None

        solver_result = self.smt_solver.solver.check(self.encoding)
        if solver_result == z3.unsat:
            self.has_assignments = False
            return None
        sat_model = self.smt_solver.solver.model()
        parameter_options = []
        for _parameter_index, var in enumerate(self.smt_solver.solver_vars):
            option = sat_model[var].as_long()
            parameter_options.append([option])

        return self.parameter_space.assume_options_copy(parameter_options)


class SmtSolver:
    def __init__(
        self,
        parameter_space: paynt.parameter_space.parameter_space.ParameterSpace,
        constraint: paynt.parameter_space.constraints.constraint.Constraint | None = None,
        colored_mdp: paynt.colored_mdp.ColoredMdp | None = None,
        task: paynt.task.SynthesisTask | None = None,
    ):

        # SMT solver containing description of the unexplored design space
        self.solver = z3.Solver()

        # for each parameter contains a corresponding solver variable. BitVec (not Int, despite CEGIS
        # never going through a z3.UserPropagateBase where that would matter) so this can share both the
        # variable-construction helper and the ConstraintContext shape with SMPMC's encoding -- see
        # paynt/parameter_space/bitvec.py and paynt/parameter_space/constraints/constraint.py.
        self.solver_vars: list[Any]
        self.solver_vars, self.width, _name_to_parameter = paynt.parameter_space.bitvec.parameter_bitvec_variables(parameter_space)
        # for each parameter contains a list of equalities [p==opt1,p==opt2,...],
        #   where p is the corresponding solver variable
        self.solver_clauses: list[list[Any]] = []

        # current depth of push/pop solving
        self.solver_depth = 0

        # create solver clauses
        for parameter in range(parameter_space.num_parameters):
            clauses = [self.create_parameter_clause(parameter, option) for option in parameter_space.parameter_options(parameter)]
            self.solver_clauses.append(clauses)

        # custom constraint (e.g. --constraint costs/prob0/prob1), built once here rather than per node:
        # the clauses reference only self.solver_vars/width, not any node-specific narrowing, and some
        # constraints (e.g. costs) are expensive enough that rebuilding per node would be wasteful
        self.constraint_clauses: list[Any] = []
        if constraint is not None:
            assert colored_mdp is not None and task is not None, "constraint requires colored_mdp and task"
            ctx = paynt.parameter_space.constraints.constraint.ConstraintContext(
                colored_mdp=colored_mdp, parameter_space=parameter_space, task=task, variables=self.solver_vars, width=self.width, viable=None
            )
            self.constraint_clauses = constraint.build(ctx)

    def create_parameter_clause(self, parameter: int, option: int) -> Any:
        var = self.solver_vars[parameter]
        return var == z3.BitVecVal(option, self.width)

    def pick_assignment(self, node: paynt.synthesizer.search_node.SearchNode) -> paynt.parameter_space.parameter_space.ParameterSpace | None:
        """:return: unexplored parameter assignment from node's parameter space (or None if no instance remains)"""
        node.encode(self)
        assert node.encoding is not None
        return node.encoding.pick_assignment()

    def pick_assignment_priority(
        self, node: paynt.synthesizer.search_node.SearchNode, priority_node: paynt.synthesizer.search_node.SearchNode | None
    ) -> paynt.parameter_space.parameter_space.ParameterSpace | None:

        if priority_node is None:
            return self.pick_assignment(node)

        # explore priority parameter subspace first
        assignment = self.pick_assignment(priority_node)
        if assignment is not None:
            return assignment

        # explore remaining members
        return self.pick_assignment(node)

    def exclude_conflicts(
        self, node: paynt.synthesizer.search_node.SearchNode, assignment: paynt.parameter_space.parameter_space.ParameterSpace, conflicts: list
    ) -> int:
        """Exclude the given conflicts from the SMT solver's search space.

        :param conflicts: a list of conflicts (may be empty)
        :return: estimate of pruned assignments
        """
        pruning_estimate = 0
        for conflict in conflicts:
            pruning_estimate += self.exclude_conflict(node, assignment, conflict)
        return pruning_estimate

    def exclude_conflict(
        self, node: paynt.synthesizer.search_node.SearchNode, assignment: paynt.parameter_space.parameter_space.ParameterSpace, conflict: list[int]
    ) -> int:
        """Exclude assignment from node's parameter space encoding using provided conflict.

        :param node: search node whose current encoding should be refined
        :param assignment: parameter assignment that yielded unsatisfiable DTMC
        :param conflict: indices of relevant parameters in the corresponding counterexample
        :return: estimate of pruned assignments
        """
        assert node.encoding is not None

        pruning_estimate = 1
        counterexample_clauses = []
        for parameter, _var in enumerate(self.solver_vars):
            if parameter in conflict:
                option = assignment.parameter_options(parameter)[0]
                counterexample_clauses.append(self.solver_clauses[parameter][option])
            else:
                if node.parameter_space.parameter_num_options(parameter) < node.parameter_space.parameter_num_options_total(parameter):
                    counterexample_clauses.append(node.encoding.parameter_clauses[parameter])
                pruning_estimate *= node.parameter_space.parameter_num_options(parameter)

        counterexample_encoding = False if len(counterexample_clauses) == 0 else z3.Not(z3.And(counterexample_clauses))
        self.solver.add(counterexample_encoding)

        return pruning_estimate

    def level(self, refinement_depth: int) -> None:
        """Reset solver depth level to correspond to refinement level."""
        if refinement_depth == 0:
            # fresh parameter_space, nothing to do
            return

        # reset to the scope of the parent (refinement_depth - 1)
        while self.solver_depth >= refinement_depth:
            self.solver.pop()
            self.solver_depth -= 1

        # create new scope
        self.solver.push()
        self.solver_depth += 1
